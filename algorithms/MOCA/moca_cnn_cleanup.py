"""MOCA (Mutually Optimal Contract Algorithm) on Clean Up.

Implements Algorithm 1 of Christoffersen et al., "Formal Contracts Mitigate Social
Dilemmas in Multi-Agent RL" (arXiv:2208.10469 / AAMAS 2023):

    for t = 1 .. (9/10) num_episodes:          # PHASE 1 -- subgame
        theta ~ P(Theta)
        train_subgame_episode(pi(s_0, theta))
    freeze pi|_{S x Theta}                     # gameplay policy frozen
    for t = 1 .. (1/10) num_episodes:          # PHASE 2 -- contracting
        theta ~ pi_i(i, 0)
        accept if all voters accept
        R <- sample_episode_reward(pi, contract)
        train_with_rewards(pi_contracting, R)

Why the two phases matter (and why this differs from a hand-designed transfer
rule): Phase 1 trains the gameplay policy on contracts drawn from a FIXED
distribution that is independent of the proposal policy. The gameplay policy
therefore learns the whole family {pi(.|s, theta)}, giving an unbiased estimate of
each agent's value V_i(s_0, theta) for every theta, rather than only for the
contracts a co-adapting proposer happened to favour. Freezing that policy before
Phase 2 means the contracting stage optimises against a fixed, already-solved
subgame -- which is what makes the selected contract a subgame-perfect equilibrium
choice rather than an artefact of joint learning dynamics.

Gameplay rewards are R'_i = R_i + theta_i(s,a) with sum_i theta_i = 0, so contracts
redistribute welfare without creating it (see contracts.py).

Only PARAMETER_SHARING=False is supported: MOCA gives every agent its own proposal
and voting policy, and the specialisation this environment is studied for (some
agents cleaning, others harvesting) requires distinct gameplay policies too.
"""
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import wandb

from algorithms.utils import checkpoint_filename, save_params, save_train_state
from algorithms.MOCA.contracts import make_contract
from algorithms.MOCA.networks import ContractActorCritic, ProposalPolicy, VotingPolicy


def episode_stats(traj_batch, num_agents):
    """Welfare / equality / transfer-volume for one batch of episodes.

    These are the outcome measures the formal-contracting results are stated in
    (the reference implementation tracks the corresponding 'transfers' and
    'transfer_equality' metrics):

    * welfare -- summed episode return across agents. Contracts are zero-sum, so
      they cannot raise welfare directly; welfare only moves if the contract
      changes BEHAVIOUR (more cleaning -> more apples). It is therefore the metric
      that says whether contracting actually mitigated the dilemma.
    * equality -- 1 - Gini over per-agent episode returns (the standard measure in
      the sequential-social-dilemma literature, e.g. Hughes et al. 2018).
    * transfer_volume -- total reward actually moved per episode. The "is the
      mechanism doing anything at all" diagnostic: zero volume means the contract
      is inert regardless of what was signed.
    """
    returns = jnp.stack([traj_batch[i].reward.sum(axis=0) for i in range(num_agents)])
    welfare = returns.sum(axis=0)                      # (NUM_ENVS,)

    # 1 - Gini, computed on the pairwise-absolute-difference form.
    diffs = jnp.abs(returns[:, None, :] - returns[None, :, :]).sum(axis=(0, 1))
    denom = 2.0 * num_agents * jnp.abs(returns).sum(axis=0) + 1e-8
    equality = 1.0 - diffs / denom

    # Zero-sum transfers: the total moved equals the sum of the positive side.
    transfers = jnp.stack(
        [traj_batch[i].info["contract_transfer"].squeeze(-1) for i in range(num_agents)]
    )                                                   # (N, NUM_STEPS, NUM_ENVS)
    transfer_volume = jnp.maximum(transfers, 0.0).sum(axis=(0, 1))

    return {
        "welfare": welfare.mean(),
        "equality": equality.mean(),
        "transfer_volume": transfer_volume.mean(),
        "returns": returns,
    }


class MOCATransition(NamedTuple):
    """PPO transition, plus the contract features the policy was conditioned on."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    contract: jnp.ndarray
    info: dict


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config["PARAMETER_SHARING"]:
        raise NotImplementedError(
            "MOCA supports PARAMETER_SHARING=False only: each agent needs its own "
            "proposal/voting policy, and role specialisation needs distinct gameplay "
            "policies. Re-run with PARAMETER_SHARING=False."
        )

    num_agents = env.num_agents
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # A contract is agreed once per EPISODE and held fixed for its duration, so one
    # PPO rollout must be exactly one episode. The env auto-resets at
    # inner_t == num_inner_steps, so that means NUM_STEPS == num_inner_steps.
    inner_steps = config["ENV_KWARGS"]["num_inner_steps"]
    if config["NUM_STEPS"] != inner_steps:
        raise ValueError(
            f"MOCA needs one rollout to be exactly one episode so a contract spans "
            f"the episode it was agreed for, but NUM_STEPS={config['NUM_STEPS']} != "
            f"num_inner_steps={inner_steps}. Set them equal."
        )

    # Split the update budget 90/10 as in Algorithm 1.
    phase1_frac = config.get("PHASE1_FRAC", 0.9)
    config["NUM_UPDATES_PHASE1"] = max(int(config["NUM_UPDATES"] * phase1_frac), 1)
    config["NUM_UPDATES_PHASE2"] = max(
        config["NUM_UPDATES"] - config["NUM_UPDATES_PHASE1"], 1
    )

    contract = make_contract(
        config.get("CONTRACT_SPACE", "cleanup"),
        num_agents,
        low=config["CONTRACT_LOW"],
        high=config["CONTRACT_HIGH"],
    )
    contract_grid = contract.grid(config["NUM_CONTRACT_BINS"])
    # Plain Python copy of the grid, purely for building metric NAMES. Formatting a
    # device array with float() fails under tracing, and label text must never depend
    # on a traced value anyway.
    contract_grid_labels = [
        contract.low + (contract.high - contract.low) * k / (config["NUM_CONTRACT_BINS"] - 1)
        for k in range(config["NUM_CONTRACT_BINS"])
    ]

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES_PHASE1"]
        )
        return config["LR"] * frac

    def train(rng):
        progress_state = {"times": {}}

        # ------------------------------------------------------------ networks
        network = [
            ContractActorCritic(env.action_space().n, activation=config["ACTIVATION"])
            for _ in range(num_agents)
        ]
        proposal_net = [ProposalPolicy(config["NUM_CONTRACT_BINS"]) for _ in range(num_agents)]
        voting_net = [VotingPolicy(num_agents) for _ in range(num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_c = jnp.zeros((1, contract.obs_dim))
        init_onehot = jnp.zeros((1, num_agents))
        init_theta = jnp.zeros((1,))

        network_params = [network[i].init(_rng, init_x, init_c) for i in range(num_agents)]
        proposal_params = [proposal_net[i].init(_rng) for i in range(num_agents)]
        voting_params = [
            voting_net[i].init(_rng, init_onehot, init_theta) for i in range(num_agents)
        ]

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        contract_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["CONTRACT_LR"], eps=1e-5),
        )

        train_state = [
            TrainState.create(apply_fn=network[i].apply, params=network_params[i], tx=tx)
            for i in range(num_agents)
        ]
        proposal_state = [
            TrainState.create(
                apply_fn=proposal_net[i].apply, params=proposal_params[i], tx=contract_tx
            )
            for i in range(num_agents)
        ]
        voting_state = [
            TrainState.create(
                apply_fn=voting_net[i].apply, params=voting_params[i], tx=contract_tx
            )
            for i in range(num_agents)
        ]

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # ------------------------------------------------- shared rollout logic
        def rollout(params_list, env_state, last_obs, theta, rng):
            """Play one full episode under contract `theta` (one value per env).

            Returns the trajectory batch plus the final env/obs state. Rewards are
            contract-augmented: R'_i = R_i + transfer_i, with transfers zero-sum
            across agents.
            """
            contract_obs = contract.to_obs(theta)  # (NUM_ENVS, contract_obs_dim)

            def _env_step(carry, unused):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
                env_act, log_prob, value = {}, [], []
                act_keys = jax.random.split(_rng, num_agents)
                for i in range(num_agents):
                    pi, value_i = network[i].apply(params_list[i], obs_batch[i], contract_obs)
                    action = pi.sample(seed=act_keys[i])
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action
                    value.append(value_i)
                env_act_list = [v for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act_list)

                # Contract transfers: zero-sum redistribution on top of base reward.
                transfers = contract.compute_transfer(theta, info["cleaned_by_agent"])
                reward = reward + transfers
                info = dict(info)
                info["contract_transfer"] = transfers
                info["contract_theta"] = jnp.broadcast_to(
                    theta[:, None], transfers.shape
                )

                done_list = [v for v in done.values()]
                transition = []
                for i in range(num_agents):
                    info_i = {
                        k: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]), 1), v[:, i])
                        for k, v in info.items()
                    }
                    transition.append(
                        MOCATransition(
                            done_list[i],
                            env_act_list[i],
                            value[i],
                            reward[:, i],
                            log_prob[i],
                            obs_batch[i],
                            contract_obs,
                            info_i,
                        )
                    )
                return (env_state, obsv, rng), transition

            (env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, (env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            return traj_batch, env_state, last_obs, rng

        def compute_gae(traj_batch_i, last_val_i):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val_i), last_val_i),
                traj_batch_i,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch_i.value

        # =================================================================
        # PHASE 1 -- learn to PLAY under contracts drawn from P(Theta)
        # =================================================================
        def _update_step_phase1(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state

            # Contract for this episode, independent of any proposal policy.
            rng, _rng = jax.random.split(rng)
            theta = contract.sample(
                _rng, (config["NUM_ENVS"],), null_prob=config["NULL_CONTRACT_PROB"]
            )

            params_list = [ts.params for ts in train_state]
            traj_batch, env_state, last_obs, rng = rollout(
                params_list, env_state, last_obs, theta, rng
            )

            # bootstrap value
            contract_obs = contract.to_obs(theta)
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = []
            for i in range(num_agents):
                _, v = network[i].apply(train_state[i].params, last_obs_batch[i], contract_obs)
                last_val.append(v)

            def _loss_fn(params, traj_batch, gae, targets, net):
                pi, value = net.apply(params, traj_batch.obs, traj_batch.contract)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(
                    ratio * gae,
                    jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                ).mean()
                entropy = pi.entropy().mean()
                total = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                return total, (value_loss, loss_actor, entropy)

            metric = []
            for i in range(num_agents):
                advantages_i, targets_i = compute_gae(traj_batch[i], last_val[i])

                def _update_epoch(update_state, unused, i=i):
                    def _update_minbatch(ts, batch_info):
                        tb, adv, tgt = batch_info
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        loss, grads = grad_fn(ts.params, tb, adv, tgt, network[i])
                        return ts.apply_gradients(grads=grads), loss

                    ts, tb, adv, tgt, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (tb, adv, tgt)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                        shuffled,
                    )
                    ts, loss_info = jax.lax.scan(_update_minbatch, ts, minibatches)
                    return (ts, tb, adv, tgt, rng), loss_info

                update_state = (train_state[i], traj_batch[i], advantages_i, targets_i, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state[i] = update_state[0]
                rng = update_state[-1]

                metric_i = dict(traj_batch[i].info)
                metric_i["loss"] = loss_info[0]
                metric_i["value_loss"] = loss_info[1][0]
                metric_i["actor_loss"] = loss_info[1][1]
                metric_i["entropy"] = loss_info[1][2]
                metric.append(metric_i)

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)

            metric = jax.tree.map(lambda x: x.mean(), metric)
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            out["contract_theta_sampled"] = theta.mean()
            # Namespaced by stage, as the reference logger does (stage_1/..., stage_2/...):
            # the two phases measure different things, so sharing a key would splice a
            # subgame-learning curve onto a contract-negotiation curve.
            out = {f"stage_1/{k}": v for k, v in out.items()}
            out["phase"] = jnp.float32(1.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_1/shaped_rewards_mean"], 1
            )

            return (train_state, env_state, last_obs, update_step, rng), out

        # =================================================================
        # PHASE 2 -- with gameplay FROZEN, learn WHICH contract to sign
        # =================================================================
        def _update_step_phase2(runner_state, unused):
            (frozen_params, proposal_state, voting_state,
             env_state, last_obs, update_step, rng) = runner_state

            # -- contracting stage: propose, then vote --
            rng, k_prop, k_idx, k_vote = jax.random.split(rng, 4)
            proposer = jax.random.randint(k_prop, (config["NUM_ENVS"],), 0, num_agents)
            proposer_onehot = jax.nn.one_hot(proposer, num_agents)

            # theta ~ pi_p(p, 0) for whichever agent p proposes in each env
            all_logits = jnp.stack(
                [proposal_net[i].apply(proposal_state[i].params).logits for i in range(num_agents)]
            )                                        # (N, K)
            sel_logits = all_logits[proposer]        # (NUM_ENVS, K)
            idx = jax.random.categorical(k_idx, sel_logits)      # (NUM_ENVS,)
            theta_prop = contract_grid[idx]
            theta_norm = (theta_prop - contract.low) / (contract.high - contract.low)

            # Unanimous consent: every non-proposer must accept. (The paper also
            # explores sampling nu voters to cut exploration variance; with only a
            # handful of agents we poll all of them, which is the rule the theory
            # is stated for.)
            vote_keys = jax.random.split(k_vote, num_agents)
            votes = []
            accept_all = jnp.ones((config["NUM_ENVS"],), dtype=bool)
            for j in range(num_agents):
                pi_j = voting_net[j].apply(voting_state[j].params, proposer_onehot, theta_norm)
                v_j = pi_j.sample(seed=vote_keys[j])     # 0 = reject, 1 = accept
                votes.append(v_j)
                is_proposer = proposer == j
                accept_j = jnp.where(is_proposer, 1, v_j)   # a proposer backs its own offer
                accept_all = accept_all & (accept_j == 1)
            votes = jnp.stack(votes)                        # (N, NUM_ENVS)

            # Rejected proposals fall back to the null contract (no transfers).
            theta_eff = jnp.where(accept_all, theta_prop, jnp.float32(contract.low))

            # -- play the episode with the FROZEN gameplay policy --
            traj_batch, env_state, last_obs, rng = rollout(
                frozen_params, env_state, last_obs, theta_eff, rng
            )
            # Episode return per agent -- the payoff the contracting policies optimise.
            returns = jnp.stack(
                [traj_batch[i].reward.sum(axis=0) for i in range(num_agents)]
            )                                                # (N, NUM_ENVS)

            # -- REINFORCE on the contracting policies --
            # One decision per episode, so this is an episode-level bandit; a
            # batch-mean baseline per agent keeps the gradient variance manageable.
            baseline = returns.mean(axis=1, keepdims=True)
            adv = returns - baseline                          # (N, NUM_ENVS)

            def proposal_loss(params, i):
                pi = proposal_net[i].apply(params)
                logp = pi.log_prob(idx)                       # (NUM_ENVS,)
                # only envs where agent i actually proposed contribute
                mask = (proposer == i).astype(jnp.float32)
                n = jnp.maximum(mask.sum(), 1.0)
                return -(logp * adv[i] * mask).sum() / n

            def voting_loss(params, j):
                pi = voting_net[j].apply(params, proposer_onehot, theta_norm)
                logp = pi.log_prob(votes[j])
                mask = (proposer != j).astype(jnp.float32)    # proposers don't vote
                n = jnp.maximum(mask.sum(), 1.0)
                return -(logp * adv[j] * mask).sum() / n

            for i in range(num_agents):
                g = jax.grad(proposal_loss)(proposal_state[i].params, i)
                proposal_state[i] = proposal_state[i].apply_gradients(grads=g)
                gv = jax.grad(voting_loss)(voting_state[i].params, i)
                voting_state[i] = voting_state[i].apply_gradients(grads=gv)

            update_step = update_step + 1
            jax.debug.callback(
                contract_checkpoint_callback, proposal_state, voting_state, update_step
            )

            metric = jax.tree.map(lambda x: x.mean(), [dict(traj_batch[i].info) for i in range(num_agents)])
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            # The headline MOCA diagnostics: what is being proposed, and is it signed?
            out["contract_theta_proposed"] = theta_prop.mean()
            out["contract_theta_effective"] = theta_eff.mean()
            out["contract_accept_rate"] = accept_all.mean()
            out["contract_returns_mean"] = returns.mean()
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            # Spread across envs is the REINFORCE signal itself: the advantage is
            # returns minus their per-agent batch mean, so if every env returns the
            # same the contracting policies get an exactly zero gradient.
            out["contract_returns_std"] = returns.std()
            out["contract_adv_absmean"] = jnp.abs(adv).mean()
            probs = jax.nn.softmax(all_logits, axis=-1).mean(axis=0)  # (K,)
            out["contract_proposal_argmax"] = contract_grid[jnp.argmax(probs)]
            # Per-bin proposal mass: the shape of this distribution over training is
            # the actual result of a MOCA run -- which contract the agents converge on.
            for k in range(config["NUM_CONTRACT_BINS"]):
                out[f"proposal_p_theta{contract_grid_labels[k]:.3f}"] = probs[k]
            out = {f"stage_2/{k}": v for k, v in out.items()}
            out["phase"] = jnp.float32(2.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_2/contract_returns_mean"], 2
            )

            return (frozen_params, proposal_state, voting_state,
                    env_state, last_obs, update_step, rng), out

        # ----------------------------------------------------------- callbacks
        def log_callback(metric):
            wandb.log({k: float(v) for k, v in metric.items()})

        def checkpoint_callback(train_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                save_params(train_state[i], f"./checkpoints/moca/{filename}_{i}.pkl")
                save_train_state(
                    train_state[i], update_step, f"./checkpoints/moca/{filename}_resume_{i}.pkl"
                )
            print(f"[checkpoint] MOCA phase-1 gameplay policy at update {update_step}")

        def contract_checkpoint_callback(proposal_state, voting_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                save_params(proposal_state[i], f"./checkpoints/moca/{filename}_proposal_{i}.pkl")
                save_params(voting_state[i], f"./checkpoints/moca/{filename}_voting_{i}.pkl")
            print(f"[checkpoint] MOCA phase-2 contracting policies at update {update_step}")

        def progress_callback(update_step, mean_val, phase):
            update_step = int(update_step)
            now = time.time()
            progress_state["times"][update_step] = now
            first = progress_state.setdefault("first_update", update_step)
            every = config.get("PROGRESS_EVERY", 1)
            if every <= 0 or update_step % every != 0:
                return
            total = config["NUM_UPDATES_PHASE1"] + config["NUM_UPDATES_PHASE2"]
            if update_step <= first:
                print(f"[progress] phase {int(phase)} update {update_step}/{total} "
                      f"(JIT compiling -- first update is slow)", flush=True)
                return
            t1 = progress_state["times"].get(first)
            if t1 is None:
                print(f"[progress] phase {int(phase)} update {update_step}/{total}", flush=True)
                return
            rate = (now - t1) / (update_step - first)
            eta = rate * (total - update_step) / 60
            print(
                f"[progress] phase {int(phase)} update {update_step}/{total} "
                f"({100*update_step/total:.1f}%) ~{rate:.1f}s/update, ETA ~{eta:.1f} min, "
                f"metric={float(mean_val):.3f}",
                flush=True,
            )

        # --------------------------------------------------------------- run
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric1 = jax.lax.scan(
            _update_step_phase1, runner_state, None, config["NUM_UPDATES_PHASE1"]
        )

        # FREEZE the gameplay policy: phase 2 only ever reads these params.
        train_state, env_state, last_obs, update_step, rng = runner_state
        frozen_params = [ts.params for ts in train_state]

        runner_state2 = (frozen_params, proposal_state, voting_state,
                         env_state, last_obs, update_step, rng)
        runner_state2, metric2 = jax.lax.scan(
            _update_step_phase2, runner_state2, None, config["NUM_UPDATES_PHASE2"]
        )

        return {
            "runner_state": (train_state,),
            "proposal_state": runner_state2[1],
            "voting_state": runner_state2[2],
            "contract_grid": contract_grid,
            "metrics_phase1": metric1,
            "metrics_phase2": metric2,
        }

    return train


SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_cleanup"}
TUNE_KWARGS = {"sweep_name": "cleanup"}
