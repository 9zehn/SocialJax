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

    if "metrics_phase2" not in out:
        # PHASE1_ONLY. The gameplay policies are the whole product of the run, so
        # print the glob that feeds them back in -- getting it wrong (catching the
        # contracting checkpoints too) is the easy mistake, and PHASE1_FROM rejects
        # it on file count rather than loading the wrong weights.
        print("\n=== phase 1 complete (PHASE1_ONLY) ===")
        print(f"  saved {num_agents} gameplay policies")
        print(f"  rerun phase 2 against them with:")
        print(f"    PHASE1_FROM='./checkpoints/moca/{filename}_[0-9].pkl'")
        return out

    if "negotiate_state" in out:
        for i in range(num_agents):
            save_params(out["negotiate_state"][i],
                        f"./checkpoints/moca/{filename}_contract_{i}.pkl")
        m = out["metrics_phase2"]
        theta = np.array(m["stage_2/contract_theta_proposed"])
        eff = np.array(m["stage_2/contract_theta_effective"])
        acc = np.array(m["stage_2/contract_accept_rate"])
        tail = max(len(theta) // 10, 1)
        print("\n=== Learned negotiation stage (agent 0 proposes) ===")
        print(f"  proposed theta (last 10%) : {theta[-tail:].mean():.4f}")
        print(f"  effective theta (last 10%): {eff[-tail:].mean():.4f}")
        print(f"  accept rate (last 10%)    : {acc[-tail:].mean():.4f}")
        return out

    if "proposal_state" not in out:
        # Solver phase 2: nothing is learned in phase 2, so the run's result is the
        # contract the search settled on, which lives in the logged metrics rather
        # than in any saved policy.
        theta = np.array(out["metrics_phase2"]["stage_2/contract_theta_effective"])
        null_rate = np.array(out["metrics_phase2"]["stage_2/solver_null_rate"])
        tail = max(len(theta) // 10, 1)
        print("\n=== Solver-selected contracts ===")
        print(f"  mean theta over phase 2 : {theta.mean():.4f}")
        print(f"  mean theta (last 10%)   : {theta[-tail:].mean():.4f}")
        print(f"  null-contract rate      : {null_rate.mean():.4f}")
        return out

    for i in range(num_agents):
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
            # Kept at/above 1e-2: total logit displacement is bounded by
            # NUM_UPDATES_PHASE2 * CONTRACT_MINIBATCHES * CONTRACT_LR, and below
            # ~1e-2 that budget is too small for the proposal policy to leave
            # uniform, so lower values sweep only degenerate runs.
            "CONTRACT_LR": {"values": [5e-2, 3e-2, 1e-2]},
            "ENT_COEF": {"values": [0.0001, 0.01, 0.1]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, entity=default_config["ENTITY"],
                           project=default_config["PROJECT"])
    wandb.agent(sweep_id, wrapped_make_train, count=100)
