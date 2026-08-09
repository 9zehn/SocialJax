"""Search candidate phase-2 protocols over a precomputed V_i(theta) table.

Every contract-selection protocol -- who proposes, who votes, what happens on
rejection -- is a DETERMINISTIC function of the table grid_eval.py produces, because
phase 2 runs against a frozen gameplay policy. So a protocol can be evaluated by
searching that table instead of training it, which turns a 3e7-step run per design
into a loop over ~10 numbers.

Use this to kill bad designs before implementing any of them in the training loop.
What it cannot tell you is how a protocol would perform against a policy trained
under it -- the table is fixed, so it answers "given how these agents play, which
contract would this protocol select", not "what would agents learn to do".

Usage:
    python algorithms/MOCA/grid_eval.py ... --save table.npz
    python algorithms/MOCA/protocols.py --table table.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

PROPOSER = 0          # MOCA's fixed proposer, and the initial proposer throughout


# --------------------------------------------------------------- primitives

def accepts(returns, k):
    """Boolean (N,): which agents strictly prefer contract k to the null contract.

    Strict, and measured against V_i(0), matching the reference solver's
    `if k1[k] > default_vals[k]`. The null contract is row 0 by construction.
    """
    return returns[k] > returns[0]


def majority_ok(returns, k):
    """Reference solver's rule: accepted >= rejected, i.e. 2*accepts >= N."""
    return 2 * accepts(returns, k).sum() >= returns.shape[1]


def unanimous_ok(returns, k, exclude=()):
    a = accepts(returns, k)
    return all(a[i] for i in range(len(a)) if i not in exclude)


def net_receivers(returns, base_returns, k):
    """Agents the contract pays more than it charges -- the cleaners it subsidises."""
    return (returns[k] - base_returns[k]) > 0


def ideal_points(returns):
    """(N,) index of each agent's most-preferred contract: what it would propose."""
    return np.argmax(returns, axis=0)


# ---------------------------------------------------------------- protocols
# Each returns an index into the theta grid, or 0 (the null contract).

def p_null(t):
    return 0


def p_utilitarian(t):
    return int(np.argmax(t["returns"].sum(axis=1)))


def p_solver_max(t):
    return p_utilitarian(t)


def p_solver_majority(t):
    """The reference implementation's default: welfare-max subject to majority IR."""
    r = t["returns"]
    ok = np.array([majority_ok(r, k) for k in range(len(r))])
    ok[0] = True                                   # null is always feasible
    return int(np.argmax(np.where(ok, r.sum(axis=1), -np.inf)))


def p_moca(t):
    """Fixed proposer takes its best contract that every other agent would sign.

    Unanimity among the non-proposers is the guaranteed-signing reading: MOCA polls
    nu=2 at random, so requiring all of them is the contract that signs whichever
    pair is drawn.
    """
    r = t["returns"]
    ok = np.array([unanimous_ok(r, k, exclude=(PROPOSER,)) for k in range(len(r))])
    ok[0] = True
    return int(np.argmax(np.where(ok, r[:, PROPOSER], -np.inf)))


def p_counteroffer_max(t):
    """User's protocol as specified: everyone counteroffers, the HIGHEST goes to a vote.

    Selecting the maximum makes proposing the ceiling a dominant strategy for anyone
    who gains from high theta -- your bid only matters when it is the maximum, and a
    harvester would never want its own high bid to win. So the vote is always
    "ceiling or null", and the counteroffer stage carries no information.
    """
    r = t["returns"]
    k = int(np.max(ideal_points(r)))
    return k if majority_ok(r, k) else 0


def p_counteroffer_median(t):
    """Median counteroffer instead of the maximum: no single extreme bid can drag it."""
    r = t["returns"]
    k = int(np.median(ideal_points(r)))
    return k if majority_ok(r, k) else 0


def _descend(t, accept_fn, delta=1.0):
    """Walk theta downward until `accept_fn` passes; the shared skeleton of the
    descending-counteroffer idea.

    `delta` is the per-round survival factor -- what a round of haggling costs
    everyone. With delta=1 delay is free.
    """
    r = t["returns"]
    order = list(range(len(r) - 1, 0, -1))         # highest theta first
    for round_idx, k in enumerate(order):
        if accept_fn(r, k, round_idx, delta):
            return k
    return 0


def p_descending_myopic(t):
    """Voters accept any offer better than the null, ignoring what rejecting would win.

    Terminates at the highest theta a majority weakly likes -- which flatters the
    mechanism, because nobody is holding out for the better offer they know is coming.
    """
    return _descend(t, lambda r, k, _i, _d: majority_ok(r, k))


def _strategic_accept(delta):
    """Voters compare accepting now against the discounted best they could hold out for.

    This is the objection to a descending protocol with free delay: if rejecting
    strictly improves your next offer and costs nothing, you reject.
    """
    def fn(r, k, round_idx, _d):
        future = [j for j in range(k - 1, 0, -1)]
        gain_now = r[k] - r[0]
        # Best future round each agent could still reach, discounted by the delay.
        best_future = np.full(r.shape[1], -np.inf)
        for step, j in enumerate(future, start=1):
            best_future = np.maximum(best_future, (delta ** step) * (r[j] - r[0]))
        if not future:
            best_future = np.zeros(r.shape[1])
        prefer_now = gain_now >= best_future
        return 2 * (prefer_now & (gain_now > 0)).sum() >= r.shape[1]
    return fn


def p_descending_strategic_nodelay(t):
    return _descend(t, _strategic_accept(1.0), delta=1.0)


def p_descending_strategic_costly(t):
    """Same, but each round of haggling costs 10% -- episode time spent negotiating."""
    return _descend(t, _strategic_accept(0.9), delta=0.9)


def p_descending_receiver_veto(t):
    """Descending offers, majority vote, plus a veto for at least one net receiver.

    The minimal guard against the cleaners -- who are the minority and are the ones
    being paid -- simply being outvoted by the harvesters funding them.
    """
    r, b = t["returns"], t["base_returns"]

    def ok(_r, k, _i, _d):
        if not majority_ok(r, k):
            return False
        recv = net_receivers(r, b, k)
        return bool(recv.any() and accepts(r, k)[recv].any())

    return _descend(t, ok)


def p_highest_acceptance(t):
    """Everyone proposes; the proposal the MOST agents would accept wins.

    This is approval voting over the agents' ideal points. Its weakness is not
    strategic -- unlike max-selection there is no dominant bid -- but distributional:
    the most widely-acceptable proposal is the one nearest the majority's preference,
    so with more harvesters than cleaners it tracks the harvesters. It also gives
    every agent an incentive to propose what is POPULAR rather than what it wants,
    since a proposal only wins by being agreeable, which collapses the proposals
    toward one point and throws away the minority's preference entirely.
    """
    r = t["returns"]
    cands = sorted(set(ideal_points(r).tolist()))
    counts = [(accepts(r, k).sum(), r[k].sum(), k) for k in cands]
    counts.sort(reverse=True)                     # most accepts, welfare as tie-break
    best = counts[0][2]
    return best if majority_ok(r, best) else 0


def p_random_dictator(t):
    """Each agent in turn is the sole proposer; report the welfare-median outcome.

    Rotating proposal rights is the obvious "make it fair by symmetry" fix. It
    equalises rights ex ante but every individual episode is still an ultimatum, so
    it trades a systematic bias for variance rather than removing the extraction.
    """
    r = t["returns"]
    picks = []
    for i in range(r.shape[1]):
        ok = np.array([unanimous_ok(r, k, exclude=(i,)) for k in range(len(r))])
        ok[0] = True
        picks.append(int(np.argmax(np.where(ok, r[:, i], -np.inf))))
    return int(np.median(picks))


def p_supermajority(t):
    """Welfare-max subject to at least 80% of agents accepting."""
    r = t["returns"]
    need = int(np.ceil(0.8 * r.shape[1]))
    ok = np.array([accepts(r, k).sum() >= need for k in range(len(r))])
    ok[0] = True
    return int(np.argmax(np.where(ok, r.sum(axis=1), -np.inf)))


def p_cleaner_veto(t):
    """Welfare-max subject to a majority AND every net receiver accepting.

    Gives the agents being paid -- the minority the contract exists to compensate --
    a hard veto, rather than letting the agents funding it outvote them.
    """
    r, b = t["returns"], t["base_returns"]
    ok = []
    for k in range(len(r)):
        recv = net_receivers(r, b, k)
        ok.append(majority_ok(r, k) and (not recv.any() or accepts(r, k)[recv].all()))
    ok = np.array(ok)
    ok[0] = True
    return int(np.argmax(np.where(ok, r.sum(axis=1), -np.inf)))


# ------------------------------------------------- bargaining-solution refs

def p_nash(t):
    r = t["returns"]
    gains = np.maximum(r - r[0], 0.0)
    return int(np.argmax(np.prod(gains, axis=1)))


def p_egalitarian(t):
    r = t["returns"]
    return int(np.argmax((r - r[0]).min(axis=1)))


def p_kalai_smorodinsky(t):
    """Contract closest to equalising each agent's fraction of its best possible gain."""
    r = t["returns"]
    gains = r - r[0]
    ideal = np.maximum(gains.max(axis=0), 1e-9)
    frac = gains / ideal
    feasible = np.array([majority_ok(r, k) for k in range(len(r))])
    feasible[0] = True
    score = np.where(feasible, frac.min(axis=1), -np.inf)
    return int(np.argmax(score))


PROTOCOLS = [
    ("null (no contract)", p_null),
    ("MOCA: fixed proposer, unanimous", p_moca),
    ("solver: welfare + majority", p_solver_majority),
    ("solver: welfare, no vote", p_solver_max),
    ("counteroffer MAX + majority", p_counteroffer_max),
    ("counteroffer MEDIAN + majority", p_counteroffer_median),
    ("descending, myopic voters", p_descending_myopic),
    ("descending, strategic, free delay", p_descending_strategic_nodelay),
    ("descending, strategic, costly delay", p_descending_strategic_costly),
    ("descending + receiver veto", p_descending_receiver_veto),
    ("all propose, highest acceptance", p_highest_acceptance),
    ("random dictator (rotating proposer)", p_random_dictator),
    ("supermajority (80%)", p_supermajority),
    ("majority + cleaner veto", p_cleaner_veto),
    ("[ref] Nash bargaining", p_nash),
    ("[ref] Kalai-Smorodinsky", p_kalai_smorodinsky),
    ("[ref] egalitarian (maximin gain)", p_egalitarian),
]


def equality(v):
    diffs = np.abs(v[:, None] - v[None, :]).sum()
    return 1.0 - diffs / (2.0 * len(v) * np.abs(v).sum() + 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", required=True, help=".npz written by grid_eval.py --save")
    args = p.parse_args()

    d = np.load(args.table)
    t = {k: d[k] for k in ("thetas", "returns", "base_returns", "cleaned")}
    r, thetas, cleaned = t["returns"], t["thetas"], t["cleaned"]
    n = r.shape[1]

    # Split agents by role so the fairness column means something. Cleaning is the
    # costly public good, so "who cleans" is the axis the contract is meant to price.
    mean_clean = cleaned.mean(axis=0)
    is_cleaner = mean_clean > mean_clean.mean()
    print(f"Roles from mean cleaning/step: cleaners={list(np.where(is_cleaner)[0])} "
          f"({np.round(mean_clean[is_cleaner], 3)}), "
          f"harvesters={list(np.where(~is_cleaner)[0])}")
    print(f"Contract grid: {thetas[0]:.3f} .. {thetas[-1]:.3f} ({len(thetas)} points)\n")

    hdr = (f"{'protocol':<38}{'theta':>7}{'welfare':>10}{'equality':>9}"
           f"{'cleaner':>9}{'harvest':>9}{'ratio':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name, fn in PROTOCOLS:
        k = fn(t)
        v = r[k]
        cl = v[is_cleaner].mean() if is_cleaner.any() else float("nan")
        hv = v[~is_cleaner].mean() if (~is_cleaner).any() else float("nan")
        print(f"{name:<38}{thetas[k]:>7.3f}{v.sum():>10.1f}{equality(v):>9.3f}"
              f"{cl:>9.1f}{hv:>9.1f}{cl / hv if hv else float('nan'):>7.2f}")


if __name__ == "__main__":
    main()
