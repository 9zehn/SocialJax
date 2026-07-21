""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import time

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
# from flax.training import checkpoints
from gymnax.wrappers.purerl import LogWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy

# Import shared network architectures
from algorithms.utils import (
    ActorCritic,
    batchify,
    batchify_dict,
    unbatchify,
    save_params,
    load_params,
    checkpoint_filename,
    save_train_state,
    load_train_state,
    evaluate_ippo as evaluate,
    Transition,
)

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # Wall-clock progress tracking (Python-side state captured by a host
        # callback below; NOT part of the jax.lax.scan carry). Scoped to this
        # train() call so repeated invocations (e.g. a hyperparameter sweep
        # calling make_train/train many times) don't share stale timings.
        progress_state = {"times": {}}

        # INIT NETWORK
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]
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
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # RESUME (optional): splice a previously-saved params+opt_state+step into
        # the freshly-created train_state(s) above, in place of the random init.
        # This is plain Python file I/O happening once at trace time (train() is
        # about to be jax.jit-ed in _runner.py), not a per-step jax operation, so
        # resumed_update_step below is a static Python int, valid as a jax.lax.scan
        # length/carry value. Restoring opt_state (not just params) matters: a
        # fresh optimizer state would reset Adam's moment estimates and the LR
        # schedule's step count to zero, silently changing training dynamics
        # instead of truly continuing from where the run left off.
        resumed_update_step = 0
        resume_from = config.get("RESUME_FROM")
        if resume_from:
            import glob
            matches = sorted(glob.glob(resume_from))
            if not matches:
                raise FileNotFoundError(f"RESUME_FROM={resume_from!r} matched no files")
            if config["PARAMETER_SHARING"]:
                if len(matches) != 1:
                    raise ValueError(
                        f"PARAMETER_SHARING=True expects exactly 1 resume file, "
                        f"got {len(matches)}: {matches}"
                    )
                loaded = load_train_state(matches[0])
                train_state = train_state.replace(
                    params=loaded["params"], opt_state=loaded["opt_state"], step=loaded["step"]
                )
                resumed_update_step = loaded["update_step"]
            else:
                if len(matches) != env.num_agents:
                    raise ValueError(
                        f"expected {env.num_agents} resume files (one per agent) for "
                        f"PARAMETER_SHARING=False, got {len(matches)}: {matches}"
                    )
                for i, m in enumerate(matches):
                    loaded = load_train_state(m)
                    train_state[i] = train_state[i].replace(
                        params=loaded["params"], opt_state=loaded["opt_state"], step=loaded["step"]
                    )
                    resumed_update_step = loaded["update_step"]  # same across agents by construction
            print(
                f"[resume] loaded {len(matches)} checkpoint(s) from {resume_from!r}, "
                f"continuing from update {resumed_update_step}/{config['NUM_UPDATES']}"
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
                
                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    for i in range(env.num_agents):
                        pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Bootstrap bonus for the pay mechanism: a small reward credited to the
                # SENDER whenever a payment actually EXECUTES (not merely attempted, so
                # this can't be farmed by spamming the pay action with no valid target),
                # annealed linearly to zero over PAY_BONUS_HORIZON updates. Exists only
                # here, in the training loop, not in the environment: annealing needs
                # update_step (how far through TRAINING we are), which the env has no
                # notion of -- it only tracks its own per-episode inner_t/outer_t, which
                # reset every episode. PAY_BONUS=0 (default) makes this an exact no-op,
                # and pay_mode="off" never adds "pay_executed" to info in the first
                # place, so this can't affect the baseline/placebo conditions.
                pay_bonus = config.get("PAY_BONUS", 0.0)
                if pay_bonus and "pay_executed" in info:
                    horizon = config.get("PAY_BONUS_HORIZON", 0)
                    frac = jnp.clip(1.0 - update_step / horizon, 0.0, 1.0) if horizon else 1.0
                    reward = reward + jnp.float32(pay_bonus) * frac * info["pay_executed"]


                if config["PARAMETER_SHARING"]:
                    info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    transition = Transition(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                        ))
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # reward_mean = jnp.mean(reward, axis=0)
                    # # reward_std = jnp.std(reward, axis=0) + 1e-8
                    # reward = (reward - reward_mean)# / reward_std

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                advantages = []
                targets = []
                for i in range(env.num_agents):
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                # if config["PARAMETER_SHARING"]:
                    
                # else:
                #     batch = jax.tree_util.tree_map(
                #         lambda x: x.reshape((batch_size,) + x.shape[2:]),  # 保持第一个维度为batch_size，自动计算第二个维度
                #         batch
                #     )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]
                
            def callback(metric):
                wandb.log(metric)

            def checkpoint_callback(train_state, update_step):
                # jax.debug.callback hands us concrete (host) values, so ordinary
                # Python control flow -- including this modulo gate -- is fine here.
                update_step = int(update_step)
                every = config.get("CHECKPOINT_EVERY", 20)
                if every <= 0 or update_step % every != 0:
                    return
                filename = checkpoint_filename(config, latest=True)
                if config["PARAMETER_SHARING"]:
                    # NB: mirrors the 'indvidual' typo in the final-save path in _runner.py,
                    # so periodic and final checkpoints land in the same directory.
                    save_params(train_state, f"./checkpoints/indvidual/{filename}.pkl")
                    # Separate file: params-only stays load_params()-compatible (used by
                    # the viewer/evaluate_ippo), _resume additionally carries opt_state
                    # and the update counter so a later RESUME_FROM can truly continue.
                    save_train_state(train_state, update_step, f"./checkpoints/indvidual/{filename}_resume.pkl")
                else:
                    for i in range(env.num_agents):
                        save_params(train_state[i], f"./checkpoints/individual/{filename}_{i}.pkl")
                        save_train_state(
                            train_state[i], update_step,
                            f"./checkpoints/individual/{filename}_resume_{i}.pkl",
                        )
                print(f"[checkpoint] saved rolling checkpoint at update {update_step}")

            def progress_callback(update_step, mean_reward):
                # Wall-clock ETA. The scan itself has no notion of time, so this is
                # purely a host-side callback using progress_state captured above.
                #
                # "First update seen in THIS run" (not literally update 1) is what
                # carries the one-off JIT compile cost -- after a RESUME_FROM, the
                # first scan iteration is update resumed_update_step+1, not 1, so
                # hardcoding "1" would never match again and the ETA would silently
                # stay stuck on the single-line fallback for the whole resumed run.
                update_step = int(update_step)
                now = time.time()
                progress_state["times"][update_step] = now
                first_update = progress_state.setdefault("first_update", update_step)

                every = config.get("PROGRESS_EVERY", 1)
                if every <= 0 or update_step % every != 0:
                    return

                total_updates = config["NUM_UPDATES"]
                if update_step <= first_update:
                    print(f"[progress] update {update_step}/{total_updates} (JIT compiling -- "
                          f"first update is slow, timing starts after this)", flush=True)
                    return

                t1 = progress_state["times"].get(first_update)
                if t1 is None:
                    print(f"[progress] update {update_step}/{total_updates}", flush=True)
                    return

                # Rate from the first-seen update -> now, so the one-off compile cost
                # doesn't pollute the ETA the way total_elapsed/total_updates would.
                rate = (now - t1) / (update_step - first_update)
                eta_min = rate * (total_updates - update_step) / 60
                pct = 100 * update_step / total_updates
                print(
                    f"[progress] update {update_step}/{total_updates} ({pct:.1f}%) "
                    f"~{rate:.1f}s/update, ETA ~{eta_min:.1f} min, "
                    f"mean_shaped_reward={float(mean_reward):.3f}",
                    flush=True,
                )

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)

            metric = jax.tree.map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                # jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                # jax.debug.callback(callback, metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]

            jax.debug.callback(callback, metric)
            jax.debug.callback(progress_callback, update_step, metric["shaped_rewards"])

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, resumed_update_step, _rng)
        # Resuming continues to the ORIGINAL target (config["NUM_UPDATES"]), it
        # doesn't add that many more on top -- e.g. resuming at update 200/780
        # runs 580 more, landing back at 780, not 980. Increase TOTAL_TIMESTEPS
        # explicitly if you want to extend beyond the original run's target.
        remaining_updates = max(config["NUM_UPDATES"] - resumed_update_step, 0)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, remaining_updates
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

# Used by algorithms/train.py to dispatch through algorithms.IPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "ippo_cnn_cleanup"}
TUNE_KWARGS       = {"sweep_name": "cleanup"}
