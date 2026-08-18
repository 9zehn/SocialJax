"""The centralised joint-control baseline: the welfare ceiling contracting aims at.

One network sees every agent's observation, chooses every agent's action, and is
paid the SUM of their rewards. There is no dilemma left in that problem -- an agent
that would free-ride is simply not modelled -- so what it converges to is (an
optimisation-limited estimate of) the best joint behaviour the environment admits.
Every decentralised arm in this project is measured against that number.

This is Christoffersen et al.'s `joint` baseline (`JointEnv` in
environments/two_stage_train.py, selected by `"joint": true` in
experiment_configs/cleanup-joint-2agents.json). They implement it as an environment
wrapper around a single nominal agent `a0`; here it is a network, which is the same
construction expressed in the form a vectorised JAX loop wants. Three properties
carry over exactly:

  * the observation is every agent's egocentric view CONCATENATED ON THE CHANNEL
    AXIS -- their `concatenated_obs`, which is what their Cleanup config uses;
  * the action is the joint action, factored into one categorical per agent;
  * the reward is `sum(env_rews.values())`, their "straightforward sum, not
    average".

What it is NOT: it is not a cooperative MARL method, and it is not a fair
comparison of learning algorithms. It removes the problem rather than solving it,
by assuming away the separate interests that make a social dilemma a dilemma. Read
it as a ceiling, and read the gap to it -- not its own level -- as the result.

Watch equality alongside welfare. Nothing here optimises for it, and the summed
objective is indifferent between one agent taking everything and an even split, so
a high-welfare ceiling can sit at an allocation no set of self-interested agents
would ever agree to. That gap is a finding rather than a defect.
"""
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import wandb

from algorithms.utils import checkpoint_filename, save_params, save_train_state
# The single source of truth for "which info field is the contracted act on this
# environment, and what is its commons". Imported from the contracting package on
# purpose: this baseline exists to be compared against those runs, so it has to
# report the same series under the same names, and duplicating the table here is
# how the two drift apart.
from algorithms.MOCA import envs as moca_envs
from algorithms.JOINT.networks import (
    CentralisedActorCritic, joint_entropy, joint_log_prob,
)


class JointTransition(NamedTuple):
    """PPO transition for the joint controller. `action` is the whole action vector."""
    done: jnp.ndarray
    action: jnp.ndarray          # (E, N)
    value: jnp.ndarray           # (E,)
    reward: jnp.ndarray          # (E,)   summed over agents
    log_prob: jnp.ndarray        # (E,)   joint
    obs: jnp.ndarray             # (E, H, W, N*C)
    info: dict


def centralised_obs(last_obs):
    """(E, N, H, W, C) -> (E, H, W, N*C): every agent's view, stacked on channels.

    The reference's `concatenated_obs`. Channel concatenation rather than a batch
    dimension is the point: the convolution sees all N views in register at every
    spatial position, so the policy can condition on where agents are relative to
    one another, which is what a central controller has over independent learners.
    """
    e, n, h, w, c = last_obs.shape
    return jnp.transpose(last_obs, (0, 2, 3, 1, 4)).reshape(e, h, w, n * c)


def episode_stats(traj_batch, per_agent_reward):
    """Welfare and equality for one batch of episodes.

    welfare is what the controller maximises, and per-agent returns are kept only
    so equality can be computed: a joint policy is free to starve an agent if that
    raises the sum, and the summed return alone cannot show it.
    """
    returns = per_agent_reward.sum(axis=0)                    # (E, N)
    welfare = returns.sum(axis=-1)                            # (E,)
    # 1 - Gini, on the pairwise-absolute-difference form, matching the contracting
    # runs so the two are directly comparable.
    diffs = jnp.abs(returns[:, :, None] - returns[:, None, :]).sum(axis=(1, 2))
    denom = 2.0 * returns.shape[-1] * jnp.abs(returns).sum(axis=-1) + 1e-8
    return {"welfare": welfare.mean(), "equality": (1.0 - diffs / denom).mean()}


def metric_names(spec):
    """The series logged to wandb. Behaviour names match the contracting runs."""
    return (
        "welfare",
        "equality",
        "returned_episode_returns_mean",
        f"{spec.progress_metric}_mean",
    ) + tuple(f"{m}_mean" for m in spec.behaviour_metrics) + (
        f"{spec.contracted_act}_std",
        # The contracted act summed over the WHOLE EPISODE and all agents -- total
        # cells cleaned, total depleting harvests, total coins stolen. The _mean
        # above is a per-agent per-step rate, which is the comparable quantity but
        # an unreadable one: 0.14 cells/agent/step is 980 tiles an episode, and only
        # one of those two numbers can be sanity-checked against the dirt spawn rate.
        f"{spec.contracted_act}_total",
        f"{spec.commons_metric}_mean",
        "loss_mean",
        "value_loss_mean",
        "entropy_mean",
    )


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    spec = moca_envs.spec_for(config["ENV_NAME"])
    moca_envs.check_reward_scale(spec, config.get("ENV_KWARGS", {}))
    METRICS = metric_names(spec)

    num_agents = env.num_agents
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # The controller is paid the sum of the per-agent rewards, so the environment
    # must hand out INDIVIDUAL rewards. Under shared_rewards=True every agent
    # already receives the summed reward, and summing again would scale welfare by
    # N -- a run that trains perfectly happily against an objective N times the
    # intended one, and whose welfare number cannot be compared with anything.
    if config["ENV_KWARGS"].get("shared_rewards", False):
        raise ValueError(
            "the joint controller sums the per-agent rewards itself, so it needs "
            "ENV_KWARGS.shared_rewards=False. Under shared_rewards=True each agent "
            "is already paid the sum and summing again multiplies welfare by "
            f"num_agents={num_agents}. Use reward=individual."
        )

    # One rollout is one episode, so `welfare` is episode welfare and lands on the
    # same axis as the contracting arms it is the ceiling for.
    inner_steps = config["ENV_KWARGS"]["num_inner_steps"]
    if config["NUM_STEPS"] != inner_steps:
        raise ValueError(
            f"NUM_STEPS={config['NUM_STEPS']} != num_inner_steps={inner_steps}. "
            f"Set them equal, so one rollout is one episode and `welfare` means "
            f"episode welfare -- the quantity every other arm reports."
        )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                      ) / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(frac, 0.0)

    def train(rng):
        progress_state = {"times": {}}

        network = CentralisedActorCritic(
            num_agents=num_agents, action_dim=env.action_space().n,
            activation=config["ACTIVATION"],
        )
        rng, _rng = jax.random.split(rng)
        obs_shape = (env.observation_space()[0]).shape
        init_x = jnp.zeros((1, obs_shape[0], obs_shape[1], obs_shape[2] * num_agents))
        params = network.init(_rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=(linear_schedule if config["ANNEAL_LR"]
                                      else config["LR"]), eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(
            jax.random.split(_rng, config["NUM_ENVS"]))

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state

            def _env_step(carry, unused):
                train_state, env_state, last_obs, rng = carry
                rng, k_act, k_step = jax.random.split(rng, 3)

                world = centralised_obs(last_obs)                 # (E, H, W, N*C)
                pi, value = network.apply(train_state.params, world)
                action = pi.sample(seed=k_act)                    # (E, N)
                log_prob = joint_log_prob(pi, action)             # (E,)

                # The action vector is sliced back into one action per agent, which
                # is `acts['a0'][i]` in the reference.
                env_act = [action[:, i] for i in range(num_agents)]
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(jax.random.split(k_step, config["NUM_ENVS"]), env_state, env_act)

                # "report straightforward sum, not average" -- the whole objective.
                joint_reward = reward.sum(axis=-1)                # (E,)
                done_flat = jnp.stack([v for v in done.values()])[0]
                return ((train_state, env_state, obsv, rng),
                        (JointTransition(done_flat, action, value, joint_reward,
                                         log_prob, world, info), reward))

            (train_state, env_state, last_obs, rng), (traj_batch, per_agent_reward) = \
                jax.lax.scan(_env_step, (train_state, env_state, last_obs, rng),
                             None, config["NUM_STEPS"])

            _, last_val = network.apply(train_state.params, centralised_obs(last_obs))

            def _calc_gae(traj_batch, last_val):
                def _body(carry, transition):
                    gae, next_value = carry
                    delta = (transition.reward
                             + config["GAMMA"] * next_value * (1 - transition.done)
                             - transition.value)
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"]
                           * (1 - transition.done) * gae)
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _body, (jnp.zeros_like(last_val), last_val), traj_batch,
                    reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calc_gae(traj_batch, last_val)

            def _loss_fn(params, tb, gae, tgt):
                pi, value = network.apply(params, tb.obs)
                log_prob = joint_log_prob(pi, tb.action)
                value_pred_clipped = tb.value + (value - tb.value).clip(
                    -config["CLIP_EPS"], config["CLIP_EPS"])
                value_loss = 0.5 * jnp.maximum(
                    jnp.square(value - tgt),
                    jnp.square(value_pred_clipped - tgt)).mean()
                ratio = jnp.exp(log_prob - tb.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(
                    ratio * gae,
                    jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                             1.0 + config["CLIP_EPS"]) * gae).mean()
                # Entropy of the JOINT policy, so the bonus is the same size
                # whatever N is -- a per-head mean would make exploration pressure
                # depend on the agent count.
                entropy = joint_entropy(pi).mean()
                total = (loss_actor + config["VF_COEF"] * value_loss
                         - config["ENT_COEF"] * entropy)
                return total, (value_loss, loss_actor, entropy)

            def _update_epoch(update_state, unused):
                def _update_minbatch(ts, batch_info):
                    tb, adv, tgt = batch_info
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(ts.params, tb, adv, tgt)
                    return ts.apply_gradients(grads=grads), loss

                ts, tb, adv, tgt, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), (tb, adv, tgt))
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled)
                ts, loss_info = jax.lax.scan(_update_minbatch, ts, minibatches)
                return (ts, tb, adv, tgt, rng), loss_info

            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                (train_state, traj_batch, advantages, targets, rng),
                None, config["UPDATE_EPOCHS"])
            train_state, rng = update_state[0], update_state[-1]

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)

            out = {}
            # info arrives as (T, E, N); average over time and envs, and keep the
            # ACROSS-AGENT spread of the contracted act, which is where a joint
            # controller's division of labour shows up.
            for k, v in traj_batch.info.items():
                v = jnp.asarray(v, dtype=jnp.float32)
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.mean(axis=(0, 1)).std()
            # Episode total of the contracted act: sum over time and agents, mean
            # over envs. Distinct from the commons series, which on Clean Up is the
            # river STOCK (cells currently not dirt) rather than a count of cleaning.
            act = jnp.asarray(traj_batch.info[spec.contracted_act], dtype=jnp.float32)
            out[f"{spec.contracted_act}_total"] = act.sum(axis=(0, 2)).mean()
            out["loss_mean"] = loss_info[0].mean()
            out["value_loss_mean"] = loss_info[1][0].mean()
            out["entropy_mean"] = loss_info[1][2].mean()
            out.update(episode_stats(traj_batch, per_agent_reward))

            missing = [k for k in METRICS if k not in out]
            if missing:
                raise KeyError(f"metrics not produced by this env: {missing}. "
                               f"Available: {sorted(out)}")
            out = {f"joint/{k}": out[k] for k in METRICS}
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(progress_callback, update_step, out["joint/welfare"])

            return (train_state, env_state, last_obs, update_step, rng), out

        def checkpoint_callback(train_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            save_params(train_state, f"./checkpoints/joint/{filename}.pkl")
            save_train_state(train_state, update_step,
                             f"./checkpoints/joint/{filename}_resume.pkl")
            print(f"[checkpoint] joint controller at update {update_step}")

        def log_callback(metric):
            wandb.log({k: float(v) for k, v in metric.items()})

        def progress_callback(update_step, welfare):
            update_step = int(update_step)
            now = time.time()
            progress_state["times"][update_step] = now
            first = progress_state.setdefault("first_update", update_step)
            every = config.get("PROGRESS_EVERY", 1)
            if every <= 0 or update_step % every != 0:
                return
            total = config["NUM_UPDATES"]
            t1 = progress_state["times"].get(first)
            if update_step <= first or t1 is None:
                print(f"[progress] update {update_step}/{total} "
                      f"(JIT compiling -- first update is slow)", flush=True)
                return
            rate = (now - t1) / (update_step - first)
            print(f"[progress] update {update_step}/{total} "
                  f"({100 * update_step / total:.1f}%) ~{rate:.1f}s/update, "
                  f"ETA ~{rate * (total - update_step) / 60:.1f} min, "
                  f"welfare={float(welfare):.1f}", flush=True)

        rng, _rng = jax.random.split(rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, (train_state, env_state, obsv, 0, _rng),
            None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train
