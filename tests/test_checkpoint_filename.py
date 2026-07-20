"""Tests for checkpoint_filename() -- the deterministic checkpoint/WandB-run naming
shared by the final save (_runner.py) and the periodic rolling save
(ippo_cnn_cleanup.py's checkpoint_callback).

Regression test for a real bug: the original naming only encoded ENV_NAME/SEED/
REWARD, so three pay_mode conditions (off/noop/on) run at the same SEED and
reward=individual -- exactly a controlled comparison's natural setup -- all
resolved to the identical filename, each run silently overwriting the last one's
checkpoint on disk with no error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.utils.io_utils import checkpoint_filename


def _config(**overrides):
    base = {
        "ENV_NAME": "clean_up",
        "SEED": 42,
        "REWARD": "individual",
        "ENV_KWARGS": {"num_agents": 4},
    }
    base.update(overrides)
    return base


def test_different_pay_modes_same_seed_do_not_collide():
    names = {
        checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_mode": mode}))
        for mode in ("off", "noop", "on")
    }
    assert len(names) == 3, f"pay_mode variants must produce distinct filenames, got {names}"


def test_different_num_agents_same_seed_do_not_collide():
    names = {
        checkpoint_filename(_config(ENV_KWARGS={"num_agents": n}))
        for n in (3, 4, 7)
    }
    assert len(names) == 3, f"num_agents variants must produce distinct filenames, got {names}"


def test_different_pay_clean_windows_same_seed_do_not_collide():
    names = {
        checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on", "pay_clean_window": w}))
        for w in (10, 50, 100)
    }
    assert len(names) == 3, f"pay_clean_window variants must produce distinct filenames, got {names}"


def test_pay_clean_window_only_marked_when_pay_active():
    # pay_clean_window is meaningless under pay_mode="off" -- must not affect the name
    off_default = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_clean_window": 50}))
    off_swept = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_clean_window": 10}))
    assert off_default == off_swept == "clean_up_seed42_reward_individual_agents4"


def test_pay_clean_window_default_matches_unmarked_naming():
    # 50 is clean_up's own default, so an explicit 50 must match omitting it entirely --
    # existing pay_on/pay_noop runs at the default window keep their current filenames.
    explicit_default = checkpoint_filename(
        _config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on", "pay_clean_window": 50})
    )
    omitted = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on"}))
    assert explicit_default == omitted == "clean_up_seed42_reward_individual_pay_on_agents4"


def test_pay_mode_off_or_absent_matches_original_naming():
    # pay_mode="off" (or omitted) shouldn't add a suffix, so old baseline runs
    # without any pay_mode key keep the same filename they always had.
    no_key = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4}))
    explicit_off = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_mode": "off"}))
    assert no_key == explicit_off == "clean_up_seed42_reward_individual_agents4"


def test_latest_flag_adds_marker_without_changing_base_name():
    final = checkpoint_filename(_config())
    rolling = checkpoint_filename(_config(), latest=True)
    assert rolling == f"{final}_latest"


def test_missing_reward_key_does_not_crash():
    # older/legacy configs may not define REWARD at all
    cfg = _config()
    del cfg["REWARD"]
    name = checkpoint_filename(cfg)
    assert "reward" not in name
    assert name == "clean_up_seed42_agents4"


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
