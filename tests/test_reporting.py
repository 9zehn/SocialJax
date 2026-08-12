"""Tests for the claims-and-audits layer (algorithms/MOCA/reporting.py).

The invariants here are the ones that would silently corrupt an experiment: a
settlement that creates or destroys reward (the mechanism must redistribute, not
subsidise), a fine that scales with the honest work rather than the crime, or a
claim policy that starts life lying (which would make "lying emerged" unreadable).

Runnable two ways:
    python tests/test_reporting.py
    python -m pytest tests/test_reporting.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.MOCA import reporting
from algorithms.MOCA.negotiate import unsquash


def _random_case(seed=0, n=7, e=16):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    theta = jax.random.uniform(k1, (e,), minval=0.0, maxval=3.0)
    overclaim = jax.random.uniform(k2, (n, e), minval=0.0, maxval=20.0)
    audited = jax.random.uniform(k3, (n, e)) < 0.3
    return theta, overclaim, audited


def test_settlement_is_zero_sum():
    """Claims redistribute reward; they must not mint or burn it."""
    theta, overclaim, audited = _random_case()
    transfer, _ = reporting.settle_claims(theta, overclaim, audited, 2.0, 7)
    np.testing.assert_allclose(np.asarray(transfer.sum(axis=0)), 0.0, atol=1e-4)


def test_own_position_pays_unaudited_and_fines_audited():
    """theta * o when the lie lands; -lam * theta * o when it is caught -- and the
    TRUE portion is never touched, so the fine scales with the crime alone."""
    theta = jnp.array([2.0])
    o = jnp.array([[5.0]])
    _, own_clean = reporting.settle_claims(
        theta, o, jnp.array([[False]]), 3.0, 7)
    _, own_caught = reporting.settle_claims(
        theta, o, jnp.array([[True]]), 3.0, 7)
    assert np.isclose(float(own_clean[0, 0]), 10.0), "unaudited pays theta*o"
    assert np.isclose(float(own_caught[0, 0]), -30.0), "audited voids AND fines lam*theta*o"


def test_null_contract_settles_nothing():
    """No contract, no payments, no fines -- whatever was claimed."""
    _, overclaim, audited = _random_case()
    transfer, own = reporting.settle_claims(
        jnp.zeros((overclaim.shape[1],)), overclaim, audited, 2.0, 7)
    assert float(jnp.abs(transfer).max()) == 0.0
    assert float(jnp.abs(own).max()) == 0.0


def test_honesty_threshold_matches_expected_value():
    """E[own] per unit overclaim is theta * (1 - p(1 + lam)): positive below the
    threshold lam* = (1-p)/p, negative above it. The experiment design leans on
    this boundary, so it is asserted rather than trusted."""
    for p in (0.1, 0.25, 0.5):
        lam_star = reporting.honesty_threshold(p)
        for lam, sign in ((lam_star * 0.8, 1.0), (lam_star * 1.25, -1.0)):
            ev = 1.0 - p * (1.0 + lam)
            assert np.sign(ev) == sign, (p, lam, ev)
    assert np.isclose(reporting.honesty_threshold(0.25), 3.0)


def test_claim_policy_starts_honest():
    """The mean action at init maps to an overclaim of exactly 0: agents must
    DISCOVER lying, not be born doing it, or the emergence result is unreadable."""
    net = reporting.ClaimPolicy()
    params = net.init(jax.random.PRNGKey(0),
                      jnp.zeros((1, reporting.FEATURE_DIM)))
    pi = net.apply(params, jnp.zeros((4, reporting.FEATURE_DIM)))
    mean_overclaim = unsquash(pi.mean()[:, 0], 0.0, 20.0)
    np.testing.assert_allclose(np.asarray(mean_overclaim), 0.0, atol=1e-5)


def test_claim_features_carry_the_contract_state():
    """The head must be able to tell 'contract in force' from 'null window' --
    under the null every claim is worthless and the bandit masks those rounds."""
    cleaning = jnp.array([[10.0, 0.0], [2.0, 5.0]])          # (N=2, E=2)
    theta = jnp.array([2.0, 0.0])
    feats = reporting.claim_features(cleaning, theta, 100, 0.0, 3.0)
    assert feats.shape == (2, 2, reporting.FEATURE_DIM)
    np.testing.assert_allclose(np.asarray(feats[0, :, 0]), [0.1, 0.0])  # scaled own
    assert float(feats[0, 0, 2]) == 1.0 and float(feats[0, 1, 2]) == 0.0  # in force
    assert float(feats[0, 1, 1]) == -1.0                     # null normalises to -1


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
