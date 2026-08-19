"""Every arm must train at a finite, non-negative learning rate.

This exists because `arm=vanilla` did not, and nothing caught it. `linear_schedule`
divided by NUM_UPDATES_PHASE1, which make_train sets to 0 for TRAINING_MODE=combined
precisely because that arm has no phase 1, so the rate was 0/0 = nan and Adam wrote
NaN into every gameplay weight on the first update. The run still logged plausible
contract series -- the negotiation policies use a CONSTANT learning rate and stayed
finite -- while gameplay was dead and welfare pinned at exactly 0.0 throughout.

Why the existing suites could not catch it: `_tiny_cfg` in test_contract_envs.py sets
ANNEAL_LR=False, so the schedule is never exercised there, and test_golden_arms.py
pins two_phase and joint, both of which were correct. So these tests run with
ANNEAL_LR=True -- the default every real run uses -- and assert on the trained
PARAMETERS rather than on a re-derivation of the schedule arithmetic, because a test
that recomputed the formula would have agreed with the bug.

Each arm trains in its own SUBPROCESS: several JAX training functions compiled back
to back in one process abort in `recursive_mutex` on macOS, unrelated to any test.
Same reason test_contract_envs.py does it.

Run: OMP_NUM_THREADS=1 PYTHONPATH=$PWD python tests/test_lr_schedule.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cfg(**over):
    """A few real updates at tiny scale, with the annealing schedule ON."""
    # Imported HERE, not at module scope: the parent process forks a child per arm,
    # and importing jax before forking is what trips the macOS recursive_mutex abort.
    from algorithms.MOCA import envs
    spec = envs.spec_for("clean_up")
    cfg = {
        "LR": 5e-4, "NUM_ENVS": 2, "NUM_STEPS": 8, "TOTAL_TIMESTEPS": 8 * 2 * 4,
        "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu", "PARAMETER_SHARING": False,
        # The whole point of this file. Every other suite turns this off.
        "ANNEAL_LR": True,
        "SEED": 0, "NUM_CONTRACT_BINS": 5, "PHASE1_FRAC": 0.5,
        "NULL_CONTRACT_FRAC": 0.5, "VOTER_SAMPLE_NU": 1, "CONTRACT_MINIBATCHES": 1,
        "CONTRACT_LR": 0.01, "NEGOTIATE_UPDATE_EPOCHS": 1, "CHECKPOINT_EVERY": 0,
        "PROGRESS_EVERY": 0, "TRAINING_MODE": "two_phase", "PHASE2_MODE": "negotiate",
        "ENV_NAME": "clean_up", "CONTRACT_SPACE": spec.contract_space,
        "CONTRACT_LOW": 0.0, "CONTRACT_HIGH": 2.0,
        "ENV_KWARGS": {"num_agents": 3, "num_inner_steps": 8, "shared_rewards": False,
                       "cnn": True, "jit": True, "apple_reward": 1.0},
    }
    cfg.update(over)
    return cfg


ARMS = {
    "moca": dict(TRAINING_MODE="two_phase", PHASE2_MODE="negotiate"),
    "vanilla": dict(TRAINING_MODE="combined", PHASE2_MODE="negotiate"),
    "renegotiate": dict(
        TRAINING_MODE="joint", PHASE2_MODE="bargain", BARGAIN_PROTOCOL="alternating",
        BARGAIN_BINDING="segment", BARGAIN_SEGMENT=4, BARGAIN_PROPOSER="rotate",
        BARGAIN_QUORUM="all", BARGAIN_FEATURES="private", BARGAIN_UPDATE_EPOCHS=1),
    "episode_lock": dict(
        TRAINING_MODE="joint", PHASE2_MODE="bargain", BARGAIN_PROTOCOL="alternating",
        BARGAIN_BINDING="episode", BARGAIN_SEGMENT=4, BARGAIN_PROPOSER="rotate",
        BARGAIN_QUORUM="all", BARGAIN_FEATURES="private", BARGAIN_UPDATE_EPOCHS=1),
}


def _run_child(arm):
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--child", arm],
        capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise AssertionError(
            f"{arm} produced no result.\nstdout tail:\n{proc.stdout[-1500:]}\n"
            f"stderr tail:\n{proc.stderr[-1500:]}")
    return json.loads(line[-1][len("RESULT "):])


def test_no_arm_trains_itself_into_nan():
    """The regression. Under the bug this passed for every arm except vanilla, whose
    gameplay parameters came out 100% NaN while its contract policies stayed clean.

    Both halves are checked from one training pass per arm: the gameplay policies,
    which use the annealed schedule, and the contracting policies, which use a
    constant rate. If only the first is NaN it is this bug; if both are, it is a
    different fault, and separating them is what makes the message worth reading.
    """
    bad = []
    for arm in ARMS:
        r = _run_child(arm)
        if r["gameplay_nan"]:
            bad.append(f"{arm}: {r['gameplay_nan']}/{r['gameplay_total']} NaN gameplay "
                       f"parameters (annealed rate)")
        for key, n in r["contract_nan"].items():
            if n:
                bad.append(f"{arm}: {n} NaN in {key} "
                           f"(constant rate -- a DIFFERENT fault)")
    assert not bad, "; ".join(bad)


def test_the_annealing_denominator_is_never_zero():
    """The denominator itself, since that is what went wrong: two_phase anneals over
    phase 1 only (phase 2 freezes the policy), every arm without a phase split over
    the whole run, and zero is never a legal denominator."""
    from algorithms.MOCA.moca_cnn import make_train
    for arm, over in ARMS.items():
        cfg = _cfg(**over)
        make_train(cfg)                       # resolves the derived update counts
        total = (cfg["NUM_UPDATES_PHASE1"] if cfg["TRAINING_MODE"] == "two_phase"
                 else cfg["NUM_UPDATES"])
        assert int(total) >= 1, f"{arm}: the schedule would divide by {total}"
        if cfg["TRAINING_MODE"] == "combined":
            assert cfg["NUM_UPDATES_PHASE1"] == 0, (
                f"{arm}: combined has no phase 1, so its phase-1 count stays 0 -- "
                f"which is exactly why it cannot be the annealing denominator")


def _child_main(arm):
    import jax, jax.numpy as jnp, wandb
    from algorithms.MOCA.moca_cnn import make_train
    wandb.init(mode="disabled")
    out = jax.jit(make_train(_cfg(**ARMS[arm])))(jax.random.PRNGKey(0))

    def count(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return (sum(int(jnp.isnan(x).sum()) for x in leaves),
                sum(int(x.size) for x in leaves))

    nan, total = count([ts.params for ts in out["runner_state"][0]])
    contract = {k: count([ts.params for ts in out[k]])[0]
                for k in ("negotiate_state", "bargain_state") if k in out}
    print("RESULT " + json.dumps({"gameplay_nan": nan, "gameplay_total": total,
                                  "contract_nan": contract}))


# Sorted, and the names are chosen so the subprocess test runs BEFORE the in-process
# one: building training functions in the parent and then forking is what trips the
# macOS recursive_mutex abort.
ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    if "--child" in sys.argv:
        _child_main(sys.argv[sys.argv.index("--child") + 1])
        sys.exit(0)
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
