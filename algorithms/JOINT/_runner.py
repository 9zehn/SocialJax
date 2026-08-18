"""JOINT runner: single_run / tune glue, mirroring algorithms/MOCA/_runner.py."""
import copy

import jax
import numpy as np
from omegaconf import OmegaConf
import wandb

from algorithms.utils import save_params, save_run_config, checkpoint_filename


def single_run(config, make_train, *, wandb_name):
    """One centralised-control run, saving the controller at the end."""
    config = OmegaConf.to_container(config)

    # Built before naming, as in MOCA: make_train fills in derived values that
    # checkpoint_filename reads, and naming first writes one run under two stems.
    train = make_train(config)
    filename = checkpoint_filename(config)

    # ./checkpoints/joint/ keeps these clear of the decentralised runs, which share
    # the same stem by construction -- same env, seed, reward and agent count, which
    # is exactly what makes them comparable and what would otherwise collide.
    sidecar = save_run_config(config, f"./checkpoints/joint/{filename}",
                              algorithm="JOINT")
    print(f"** Run config -> {sidecar} **")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        # Not a contracting run and not an independent-learner run: one policy, one
        # summed reward. Tagged so it cannot be read as either.
        tags=["JOINT", "CENTRALISED", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config.get("WANDB_RUN_NAME") or f"{wandb_name}_{filename}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    out = jax.jit(train)(rng)

    print("** Saving Results **")
    save_params(out["runner_state"][0], f"./checkpoints/joint/{filename}.pkl")

    m = out["metrics"]
    welfare = np.array(m["joint/welfare"])
    equality = np.array(m["joint/equality"])
    tail = max(len(welfare) // 10, 1)
    print("\n=== Centralised joint control (last 10%) ===")
    print(f"  welfare  : {welfare[-tail:].mean():.1f}")
    print(f"  equality : {equality[-tail:].mean():.3f}")
    print("  This is a CEILING, not a comparison: the controller is handed every")
    print("  agent's observation and paid their summed reward, so no dilemma")
    print("  remains. Report the GAP between a decentralised arm and this number.")
    if equality[-tail:].mean() < 0.8:
        # Worth saying out loud. The summed objective is indifferent between an
        # even split and one agent taking everything, so a ceiling can sit at an
        # allocation no self-interested agent would ever sign up to -- which makes
        # it an unreachable target for any voluntary mechanism, not just a hard one.
        print("  [note] the ceiling is reached at an UNEQUAL allocation. A "
              "voluntary mechanism cannot be expected to match a welfare number "
              "that individual rationality rules out; compare against the "
              "equality series too.")
    return out


def tune(default_config, make_train, *, sweep_name):
    """Hyperparameter sweep with wandb."""
    default_config = OmegaConf.to_container(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        print("running experiment with params:", config)
        jax.jit(make_train(config))(jax.random.PRNGKey(config["SEED"]))

    sweep_config = {
        "name": f"joint_cnn_{sweep_name}",
        "method": "bayes",
        "metric": {"name": "joint/welfare", "goal": "maximize"},
        "parameters": {
            "LR": {"values": [1e-3, 5e-4, 1e-4]},
            "ENT_COEF": {"values": [0.0001, 0.01, 0.1]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, entity=default_config["ENTITY"],
                           project=default_config["PROJECT"])
    wandb.agent(sweep_id, wrapped_make_train, count=100)
