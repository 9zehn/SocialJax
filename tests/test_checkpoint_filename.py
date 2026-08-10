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


def test_tithe_scheme_marked_and_does_not_collide_with_instant():
    instant = checkpoint_filename(_config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on"}))
    tithe = checkpoint_filename(
        _config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on", "pay_scheme": "tithe"})
    )
    assert instant != tithe, "tithe and instant runs at the same seed must not collide"
    assert tithe == "clean_up_seed42_reward_individual_pay_on_tithe_agents4"
    # "instant" is the default scheme: explicit instant matches omitting it, so all
    # existing instant-scheme checkpoints keep their current filenames
    explicit_instant = checkpoint_filename(
        _config(ENV_KWARGS={"num_agents": 4, "pay_mode": "on", "pay_scheme": "instant"})
    )
    assert explicit_instant == instant


def test_tithe_fraction_and_duration_marked_only_off_default():
    base = {"num_agents": 4, "pay_mode": "on", "pay_scheme": "tithe"}
    default = checkpoint_filename(_config(ENV_KWARGS={**base, "share_fraction": 0.5, "share_duration": 50}))
    assert default == "clean_up_seed42_reward_individual_pay_on_tithe_agents4"
    swept = checkpoint_filename(_config(ENV_KWARGS={**base, "share_fraction": 0.25, "share_duration": 100}))
    assert swept == "clean_up_seed42_reward_individual_pay_on_tithe_f0.25_d100_agents4"
    # sweeping either knob alone must also produce distinct names
    names = {
        default,
        checkpoint_filename(_config(ENV_KWARGS={**base, "share_fraction": 0.25})),
        checkpoint_filename(_config(ENV_KWARGS={**base, "share_duration": 100})),
    }
    assert len(names) == 3


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


def test_moca_phase2_modes_do_not_collide():
    """The three phase-2 modes are meant to be compared at one seed and reward,
    so they must not resolve to the same checkpoint path."""
    names = set()
    for mode in ("negotiate", "solver", "reinforce"):
        cfg = _config()
        cfg["PHASE2_MODE"] = mode
        name = checkpoint_filename(cfg)
        assert mode in name, f"{mode} not marked in {name}"
        names.add(name)
    assert len(names) == 3, f"phase-2 modes collided: {names}"


def test_non_moca_configs_are_unaffected_by_the_phase2_marker():
    """PHASE2_MODE exists only in MOCA configs; every other algorithm's filenames
    must be byte-identical to before it was introduced."""
    cfg = _config()
    assert "PHASE2_MODE" not in cfg
    assert checkpoint_filename(cfg) == "clean_up_seed42_reward_individual_agents4"


def test_negotiate_nu_is_in_the_checkpoint_name():
    """nu changes who gates the contract and so what the negotiation policy learns;
    a nu=2 and a nu=all run at one seed must not share a path."""
    names = set()
    for nu in (2, 6):
        cfg = _config()
        cfg["PHASE2_MODE"] = "negotiate"
        cfg["NEGOTIATE_NU"] = nu
        name = checkpoint_filename(cfg)
        assert f"nu{nu}" in name, name
        names.add(name)
    assert len(names) == 2, names


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
