"""Shared IPPO runner: single_run / tune glue, factored out of the 9 per-env files.

Each algorithms/IPPO/ippo_cnn_<env>.py defines its own `make_train(config)` (the
training loop, unchanged from the original per-env code) and then delegates to
single_run() or tune() here, passing env-specific strings as kwargs:

    @hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_coins")
    def main(config):
        if config["TUNE"]:
            tune(config, make_train, sweep_name="coins")
        else:
            single_run(config, make_train, wandb_name="ippo_cnn_coins")
"""
import copy

import jax
from omegaconf import OmegaConf
import wandb

import socialjax
from algorithms.utils import (save_params, load_params, checkpoint_filename,
                              save_run_config, evaluate_ippo as evaluate)


def single_run(config, make_train, *, wandb_name):
    """One training run, saving + evaluating at the end."""
    config = OmegaConf.to_container(config)

    # checkpoint_filename encodes SEED/REWARD/pay_mode/num_agents so two runs that
    # differ in any of those (e.g. your 3 pay_mode conditions at the same SEED)
    # can't silently collide onto the same checkpoint path -- also used for the
    # WandB run name so those don't collide/get confused in the dashboard either.
    filename = checkpoint_filename(config)

    # Resolved config beside the checkpoints, written before training so a run
    # killed partway still leaves usable provenance. Env kwargs that change what
    # a baseline MEANS (shared_rewards, initial_dirt_fraction, apple_reward) are
    # not all in the filename, and a replay at the wrong ones is silently wrong.
    save_run_config(config, f"./checkpoints/individual/{filename}", algorithm="IPPO")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        # WANDB_RUN_NAME overrides the auto-generated name for this one run (e.g. a
        # short label like "baseline_seed42") without affecting checkpoint filenames,
        # which always come from checkpoint_filename() regardless of this setting.
        name=config.get("WANDB_RUN_NAME") or f"{wandb_name}_{filename}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/individual/{filename}.pkl"
    if config["PARAMETER_SHARING"]:
        # NB: original code had this 'indvidual' typo, preserved here.
        save_path = f"./checkpoints/indvidual/{filename}.pkl"
        save_params(train_state, save_path)
        params = load_params(save_path)
    else:
        params = []
        for i in range(config['ENV_KWARGS']['num_agents']):
            save_path = f"./checkpoints/individual/{filename}_{i}.pkl"
            save_params(train_state[i], save_path)
            params.append(load_params(save_path))

    if config.get("EVALUATE", True):
        evaluate(params, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)
    else:
        print("EVALUATE=False -- skipping post-training rollout/GIF render")


def tune(default_config, make_train, *, sweep_name):
    """Hyperparameter sweep with wandb."""
    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": sweep_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            # "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
            "SEED": {"values": [42, 52, 62]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree.map(lambda x: x[0], outs["runner_state"][0])

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)
