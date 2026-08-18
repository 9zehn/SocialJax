"""Smoke test: does the actual training entry point import cleanly?

tests/test_pay_mechanism.py exercises the environment directly and never touches
algorithms/utils/networks.py (where `distrax`/`tensorflow_probability` get pulled
in), so it can pass while `python algorithms/train.py ...` still fails at import
time -- exactly what happened on a fresh Colab runtime whose preinstalled JAX was
newer than this repo's pinned dependencies expect (distrax's TFP-jax substrate
called a JAX internal removed in jax 0.7.0).

This test imports the same module chain algorithms/train.py does, so a dependency-
version mismatch shows up here in ~2 seconds instead of after launching a real run.
Run this immediately after any fresh environment setup (new conda env, new Colab
runtime, etc.), before running tests/test_pay_mechanism.py or any training command.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_networks_module_imports():
    import algorithms.utils.networks  # noqa: F401  (pulls in distrax/flax)


def test_ippo_cleanup_entry_point_imports():
    import algorithms.IPPO.ippo_cnn_cleanup  # noqa: F401  (same chain algorithms/train.py uses)


def test_every_contracting_entry_point_imports():
    """algorithms/train.py resolves `--env <stem>` to algorithms.MOCA.moca_cnn_<stem>,
    so a per-env entry point that does not import is a run that dies at launch --
    and these three are thin modules whose whole job is to be importable under that
    name and expose make_train."""
    import importlib

    for stem in ("cleanup", "harvest", "coins"):
        mod = importlib.import_module(f"algorithms.MOCA.moca_cnn_{stem}")
        assert callable(mod.make_train), stem
        assert mod.SINGLE_RUN_KWARGS["wandb_name"] == f"moca_cnn_{stem}"


def test_joint_control_entry_points_import():
    """The welfare ceiling every contracting number is stated against."""
    import importlib

    for stem in ("cleanup", "harvest", "coins"):
        mod = importlib.import_module(f"algorithms.JOINT.joint_cnn_{stem}")
        assert callable(mod.make_train), stem
        assert mod.SINGLE_RUN_KWARGS["wandb_name"] == f"joint_cnn_{stem}"


def test_train_py_resolves_every_registered_algo():
    """--algo X --env Y resolves to algorithms/X/<prefix>_cnn_Y.py, so a family
    registered without its modules is a run that dies at launch."""
    import importlib
    from algorithms.train import ALGO_PREFIX

    assert ALGO_PREFIX["JOINT"] == "joint"
    for algo, prefix in (("MOCA", "moca"), ("JOINT", "joint")):
        importlib.import_module(f"algorithms.{algo}._runner")
        for env in ("cleanup", "harvest", "coins"):
            importlib.import_module(f"algorithms.{algo}.{prefix}_cnn_{env}")


def test_jax_and_distrax_are_compatible():
    import jax
    import distrax

    pi = distrax.Categorical(logits=jax.numpy.zeros((2, 4)))
    sample = pi.sample(seed=jax.random.PRNGKey(0))
    assert sample.shape == (2,)


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
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
