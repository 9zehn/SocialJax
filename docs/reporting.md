# Claims and audits: imperfect enforcement (`algorithms/MOCA/reporting.py`)

Everywhere else in this codebase, contract transfers are perfectly enforced: θ per
cell *actually* cleaned, read off the environment's ground truth. That assumes away
the enforcement problem that makes real contracts hard — work is observed by the
worker, not the counterparty. This module relaxes that constraint in the smallest
strategic step that still contains the whole question.

## The mechanism (rung 1: exogenous audits)

Each window (one `BARGAIN_SEGMENT`), on top of the per-step true payment:

- every agent files an **overclaim** oᵢ ∈ [0, `REPORT_MAX_OVERCLAIM`] on its
  cleaning, paid at θ·oᵢ and funded evenly by the other N−1 agents. The claim is
  parameterised as truth-plus-overclaim because underclaiming is strictly
  dominated — letting agents report below truth adds a learning burden with zero
  strategic content;
- with probability `REPORT_AUDIT_P` the claim is **audited** against ground truth:
  the overclaim payment is voided and a fine of `REPORT_FINE_MULT`·θ·oᵢ is levied,
  flowing back to the group through the same even-funding convention (the whole
  settlement is exactly zero-sum, asserted in `test_reporting.py`);
- the **true portion is never forfeited**. The fine scales with the crime, not
  with how much honest work the liar happened to do — forfeiture would punish big
  cleaners hardest for the same lie and contaminate the cleaner/harvester
  comparison.

The expected value of one unit of overclaim is θ·(1 − p·(1+λ)), so **honesty wins
exactly when λ > λ\* = (1−p)/p** (Becker's calculus; Townsend's costly state
verification). The experiment is whether learning finds that boundary. The yaml
default (p=0.25, λ=2 < λ\*=3) sits on the *lying* side: the first question is
whether over-reporting emerges at all. The follow-up is a (p, λ) grid across λ\*.

Because audits are exogenous coin flips, the claim is a **contextual bandit** —
settlement lands immediately, nothing carries over — trained by plain REINFORCE
against a batch-standardised baseline (`claim_loss`). No semi-MDP machinery. The
settlement lands on the window's last step like any transfer, so it flows into
gameplay returns, the bargaining round rewards (negotiating agents feel the
enforcement leakage when choosing θ) and every welfare metric through the one
existing reward path. Note welfare itself is unmoved by lying (transfers are
zero-sum); the damage is distributional, and — once audits cost something at
rung 2 — real.

Claim policies start **honest by construction**: the mean action at init maps to
an overclaim of exactly 0, so lying must be discovered through exploration, and
"over-reporting emerged" is readable off the `joint/report/overclaim` series.

## Running and evaluating

```bash
python algorithms/train.py --algo MOCA --env cleanup reward=individual \
  TRAINING_MODE=joint PHASE2_MODE=bargain \
  REPORT_ENABLE=true REPORT_AUDIT_P=0.25 REPORT_FINE_MULT=2.0 \
  ... (bargaining flags as in bargaining.md)
```

Training writes `<stem>_claim_<i>.pkl` beside the other checkpoints, and the
sidecar records the REPORT_* parameters. `evaluate_bargain.py` detects the claim
checkpoints automatically, replays the enforcement layer with the run's own
(p, λ, cap) — refusing to guess if the sidecar is missing and no flags are given —
and prints a *claims and audits* block: mean overclaim against the λ\* boundary,
leakage and fines per episode, and per-agent overclaim vs true cleaning (do
cleaners or harvesters lie more? — cleaners have cover, harvesters have nothing
to audit against).

wandb series: `joint/report/overclaim` (mean chosen overclaim per claim) and
`joint/report/leakage` (overclaimed reward actually paid per episode). Both are
always logged, zero when `REPORT_ENABLE` is off.

The viewer does not replay claims; use it for gameplay/bargaining only on
reporting runs.

## Rung 2, held in reserve

Endogenous audits — harvesters choose to audit at cost κ, fines flow to the
auditor — turn the coin flip into an inspection game (Avenhaus) whose equilibrium
audit rate must sustain honesty by itself. One more Bernoulli head plus a routing
change in `settle_claims`. Not implemented; do not add reputation, per-tile
claims, or spatial auditing before rung 2 has run.
