"""MOCA runner: single_run / tune glue, mirroring algorithms/IPPO/_runner.py."""
import copy

import jax
import numpy as np
from omegaconf import OmegaConf
import wandb

from algorithms.utils import save_params, checkpoint_filename


def single_run(config, make_train, *, wandb_name):
    """One MOCA training run (phase 1 then phase 2), saving all policies at the end."""
    config = OmegaConf.to_container(config)

    filename = checkpoint_filename(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MOCA", "CONTRACTS", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config.get("WANDB_RUN_NAME") or f"{wandb_name}_{filename}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    print("** Saving Results **")
    num_agents = config["ENV_KWARGS"]["num_agents"]
    train_state = out["runner_state"][0]
    for i in range(num_agents):
        save_params(train_state[i], f"./checkpoints/moca/{filename}_{i}.pkl")
        save_params(out["proposal_state"][i], f"./checkpoints/moca/{filename}_proposal_{i}.pkl")
        save_params(out["voting_state"][i], f"./checkpoints/moca/{filename}_voting_{i}.pkl")

    # The learned contract distribution is the headline result of a MOCA run: which
    # contract the agents actually converge on proposing, and how often it is signed.
    grid = np.array(out["contract_grid"])
    logits = np.stack([
        np.array(out["proposal_state"][i].params["params"]["proposal_logits"])
        for i in range(num_agents)
    ])
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    print("\n=== Learned contract proposals (theta -> mean probability) ===")
    for k, theta in enumerate(grid):
        print(f"  theta={theta:.4f}   p={probs[:, k].mean():.4f}")
    print(f"\nModal proposed contract: theta={grid[int(probs.mean(axis=0).argmax())]:.4f}")

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
        rng = jax.random.PRNGKey(config["SEED"])
        jax.jit(make_train(config))(rng)

    sweep_config = {
        "name": f"moca_cnn_{sweep_name}",
        "method": "bayes",
        "metric": {"name": "contract_returns_mean", "goal": "maximize"},
        "parameters": {
            "LR": {"values": [1e-3, 5e-4, 1e-4]},
            "CONTRACT_LR": {"values": [1e-2, 3e-3, 1e-3]},
            "ENT_COEF": {"values": [0.0001, 0.01, 0.1]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, entity=default_config["ENTITY"],
                           project=default_config["PROJECT"])
    wandb.agent(sweep_id, wrapped_make_train, count=100)
