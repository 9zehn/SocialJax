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

    # Build the training function BEFORE naming the run. make_train fills in derived
    # config values (NEGOTIATE_NU, NUM_UPDATES_*), and checkpoint_filename reads some
    # of them -- naming first gave the final checkpoints a different stem from the
    # rolling _latest ones written from inside the run, so one run appeared on disk
    # twice under two names (..._negotiate_N.pkl and ..._negotiate_nu2_latest_N.pkl).
    # It also means wandb now records the RESOLVED config rather than the raw one.
    train = make_train(config)
    filename = checkpoint_filename(config)

    # "--algo MOCA" only selects this directory; it does not mean the run uses MOCA's
    # two-phase algorithm. Under TRAINING_MODE=joint there is no phase split, no
    # frozen subgame policy and no P(Theta), so tagging it MOCA would mislabel the
    # one axis these experiments are comparing.
    joint = config.get("TRAINING_MODE", "two_phase") == "joint"
    tags = ["CONTRACTS", "FF", "JOINT" if joint else "MOCA"]
    protocol = config.get("PHASE2_MODE")
    if protocol:
        tags.append(protocol.upper())

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
        name=config.get("WANDB_RUN_NAME") or f"{wandb_name}_{filename}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    out = jax.jit(train)(rng)

    print("** Saving Results **")
    num_agents = config["ENV_KWARGS"]["num_agents"]
    train_state = out["runner_state"][0]
    for i in range(num_agents):
        save_params(train_state[i], f"./checkpoints/moca/{filename}_{i}.pkl")

    if "metrics_joint" in out:
        # TRAINING_MODE=joint: no phase split, so the run's product is BOTH the
        # gameplay policies and the bargaining policies.
        for i in range(num_agents):
            save_params(out["bargain_state"][i],
                        f"./checkpoints/moca/{filename}_contract_{i}.pkl")
        m = out["metrics_joint"]
        tail = max(len(np.array(m["joint/outcome/welfare"])) // 10, 1)

        def last(key):
            return float(np.array(m[key])[-tail:].mean())

        print("\n=== Rubinstein bargaining, joint training (last 10%) ===")
        print(f"  agreement rate       : {last('joint/agree/rate'):.3f}")
        print(f"  agreement round      : {last('joint/agree/round'):.2f} "
              f"of {config['BARGAIN_ROUNDS']}")
        print(f"  steps lost to delay  : "
              f"{last('joint/agree/round') * config['BARGAIN_SEGMENT']:.0f}")
        print(f"  theta agreed         : {last('joint/contract/theta_agreed'):.4f}")
        print(f"  contract in force    : {last('joint/contract/in_force_rate'):.3f}")
        print(f"  welfare / equality   : {last('joint/outcome/welfare'):.1f} / "
              f"{last('joint/outcome/equality'):.3f}")
        if last("joint/contract/in_force_rate") < 0.02:
            # The predictable failure of dropping MOCA: early on the gameplay policy
            # cannot clean, so a contract really is worthless and rational agents
            # reject everything -- after which the bargaining policy never observes a
            # contract in force and has nothing to learn from.
            print("  [warning] a contract was almost never in force. This is the "
                  "cold-start failure, not a bug: raise BARGAIN_ACCEPT_BIAS or "
                  "BARGAIN_ENT_COEF, or start from BARGAIN_QUORUM=majority.")
        return out

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
