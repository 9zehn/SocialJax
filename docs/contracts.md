# Formal contracting in Clean Up (`algorithms/MOCA/`)

Reimplementation of Christoffersen, Haupt et al., *Formal Contracts Mitigate Social
Dilemmas in Multi-Agent RL* (arXiv:2208.10469 / AAMAS 2023), plus the extensions this
dissertation adds. Reference code:
`github.com/Algorithmic-Alignment-Lab/contracts`.

Configuration lives in [../algorithms/MOCA/config/moca_base.yaml](../algorithms/MOCA/config/moca_base.yaml),
which carries the per-knob rationale. This file is the map, not the manual.
Bargaining has its own page: [bargaining.md](bargaining.md).

## The mechanism

A contract is a function θ: Ω → ℝ^N from observable outcomes to a **zero-sum**
vector of reward transfers. Agents play the base game with R'_i = R_i + θ_i, so a
contract cannot create or destroy welfare — it only redistributes.

For Clean Up the space is one scalar: **θ = payment per waste cell cleaned, funded
evenly by the other agents.**

```
receive_i  = θ · c_i                          # paid for what you cleaned
pay_i      = θ · (Σ_j c_j − c_i) / (N − 1)    # you fund everyone else's
transfer_i = receive_i − pay_i                # Σ_i transfer_i = 0
```

A cleaner is a net receiver, a pure harvester a net payer. The contract subsidises
exactly the under-provided public good at the expense of those free-riding on it.

The env signal is `info["cleaned_by_agent"]` — which is why the edge-of-grid
cleaning bug (`e2e799e`) fed straight into transfers, and why every pre-fix number
is void. See [findings.md](findings.md).

## The contract space is {0} ∪ [low, high]

The null contract is **separate from the range floor**, not equal to it. With
`CONTRACT_LOW > 0`, using the floor as "no contract" would mean the fallback itself
moved reward, silently corrupting the disagreement value V_i(s, 0) that every
acceptance rule compares against.

Keeping them separate lets the range exclude weak contracts — θ below ~0.2 is too
small to change behaviour against a unit apple, and those samples only blur the
boundary between contracted and uncontracted play — while θ=0 stays genuinely null.

**The range is not recorded in the checkpoint filename.** It is written to each run's
`.run.yaml` sidecar at run start, and the viewer and evaluation tools read it from
there. Runs predating the sidecar have no record; the tools warn. This matters more
than it sounds — see the trap list in `CLAUDE.md`.

## Contract observation: `[θ_norm, is_null, stage]`

θ is normalised onto [−1, 1] over [low, high]. `is_null` is **+1 / −1**, not 1 / 0.
`stage` is the reference's contract-state indicator (0 subgame, 2 propose, 3 agree).

The `is_null` flag is a deliberate deviation and is load-bearing. The contract vector
is concatenated onto the CNN embedding before a Dense layer, whose weight gradient is
(upstream grad) ⊗ (input). The reference encoding puts the null contract at exactly
the zero vector, so the contract pathway contributes nothing forward and receives
nothing backward at θ=0. Null episodes could then only move the shared trunk and
biases, collapsing the whole θ-response to a rank-1 additive shift *continuous in θ*
— which forces behaviour at θ=0 to be the limit of behaviour at θ=ε.

That is the wrong inductive bias. "No contract" and "contract in force" are
qualitatively different regimes, not two points on a ramp. The ±1 coding (rather than
1/0) also keeps the feature vector non-zero at the *midpoint* of the contracted
range, which a 0/1 flag would reintroduce.

Empirically this worked: see the 2026-08-10 entry in [findings.md](findings.md).

Pre-fix checkpoints have a 2-feature observation and need
`LegacyCleanupContract`; `contract_for_params()` picks the right one by reading the
first Dense layer's width off the weights, so a wrong guess is an error rather than a
silently wrong number.

## Algorithm 1: two phases

| phase | what learns | contract |
|---|---|---|
| 1 (`PHASE1_FRAC` = 0.9) | contract-conditioned gameplay policy | drawn from fixed P(Θ) |
| 2 | the contracting stage | chosen against a **frozen** phase-1 policy |

P(Θ) must be fixed, policy-independent and full-support for V_i(s₀, θ) to be
unbiased across the space — which is what makes the phase-2 choice subgame-perfect.
It does **not** need to be uniform, which is why `NULL_CONTRACT_FRAC` may be raised
to reallocate estimation accuracy toward the disagreement point.
`sample_batch()` draws an exact null block plus a stratified remainder rather than
i.i.d., because at 128 envs an i.i.d. 10% varies by ±3.4 envs per update — noise on
the one quantity everything downstream is measured against.

**Workflow for a controlled protocol comparison.** The decision rule never touches
phase 1, so every arm must share *one* frozen policy rather than retraining an
identical phase 1 per arm:

```bash
# once
python algorithms/train.py --algo MOCA +PHASE1_ONLY=True
# then one cheap run per arm
python algorithms/train.py --algo MOCA \
  PHASE2_MODE=solver SOLVER_DECISION_RULE=nash \
  PHASE1_FROM='./checkpoints/moca/<stem>_[0-9].pkl'
```

`PHASE1_FROM` rejects a glob matching the wrong file count rather than loading the
wrong weights — the easy mistake is catching the contracting checkpoints too.

## Phase-2 arms (`PHASE2_MODE`)

- **`negotiate`** — MOCA's contracting game, and **the baseline the papers report**.
  Agent 0 always proposes; the action is a continuous Box `[θ, accept_prob]`; ν
  sampled non-proposers gate the contract by the **product** of their accept
  probabilities. One take-it-or-leave-it offer, no counteroffers. ν=2 is the paper's
  recommendation; do not set ν=N−1, since a product of 6 near-even probabilities is
  1.6% and the proposal policy then never sees a contract in force.
- **`solver`** — sampling-based `NegotiationSolver`: no proposer, contracts scored by
  the frozen critic and chosen by a decision rule. **Appears in neither paper
  version** — only in the code release (four months after AAMAS 2023), where every
  shipped config enables it and thereby disables the learned stage. Decision rules
  here extend the reference's `majority` with `nash`, `kalai_smorodinsky` and
  `egalitarian`; the three bargaining solutions require unanimous participation,
  where `majority` lets a contract pass over a minority's objection.
- **`reinforce`** — this repo's discretised proposal/voting game over a θ grid (null
  always at index 0). Kept so the three are comparable.
- **`bargain`** — alternating offers. See [bargaining.md](bargaining.md).

## Deviations from the reference, and why

1. **`is_null` flag** in the contract observation (above).
2. **Null contract separate from the range floor**, so the range can exclude weak
   contracts.
3. **Stratified, exact-count null sampling** instead of i.i.d. `null_prob`.
4. **θ range raised** past the paper's [0, 0.2]. That range was calibrated to
   upstream's 0.5 dirt/step; under this repo's ecology the negotiated contract pinned
   to the 0.2 ceiling — a boundary solution, meaning the reported θ measured the
   limit of the range rather than what agents wanted. Overpaying cleaners makes the
   harvesters funding it worse off than the null, and they reject, so the range only
   needs to *contain* the optimum.
5. **Extra solver decision rules** (Nash, Kalai–Smorodinsky, egalitarian).
6. **Alternating offers** as a fourth arm.

## Evaluating

```bash
# V_i(s_0, theta) over a theta sweep, for a frozen phase-1 policy
python algorithms/MOCA/grid_eval.py --checkpoint '<stem>_[0-9].pkl'

# like-for-like arm comparison; contract ranges come from each run's sidecar
python algorithms/MOCA/compare_arms.py \
  --run "moca42=runs/moca_baseline/7agents/seed42/..._negotiate_nu2_[0-9].pkl" \
  --null --episodes 20 --num-steps 1000
```

Compare arms **at θ=0**, which means the same thing everywhere — not at θ=max, since
the ranges differ between runs.

## Known issues

- Inequality between specialised cleaners and harvesters is a **protocol** problem,
  not a contract-space problem: the space spans equality (parity at θ≈0.87), but the
  one-shot ultimatum lets the proposer extract to the point of indifference
  (Proposition 4.5).
- Cleaning throughput saturates at the 1.0 cells/step dirt spawn rate, so a flat
  clean/step across θ is a conservation cap, not an unresponsive mechanism. Report
  river stock or welfare alongside it.
