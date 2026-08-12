# Findings log

Append-only. Each entry states what was measured, and carries a status:

- **STANDS** — still believed, still quotable.
- **RETRACTED** — the numbers are wrong. The entry stays, with the reason.
- **SUPERSEDED** — measured again under better conditions; see the later entry.

Never edit a number in place. A retraction that leaves no trace is how a
retracted figure ends up back in the thesis. Checkpoint provenance lives in
[../runs/INDEX.md](../runs/INDEX.md); mechanism descriptions in
[contracts.md](contracts.md), [bargaining.md](bargaining.md),
[monetary_system.md](monetary_system.md).

---

## 2026-07-22 — the dilemma is real, and 4 agents cannot escape it

**STANDS.** Baseline IPPO reproduces the expected failure: `waste_cleared` decays
over training as agents free-ride. The dilemma is present in the setup, not
assumed.

**STANDS.** Under the income-coupled tithe, payment volume stayed near zero
across the 7-agent runs. Left unresolved between: the ecology fix not yet run
long enough, the tithe redesign needing longer to be discovered, or the bootstrap
bonus needing tuning. Not revisited since the strand moved to formal contracting.

## 2026-08-10 — phase-1 null-contract conditioning

The problem: agents did not behave differently under θ=0, so the disagreement
point V_i(s, 0) was contaminated by cooperative behaviour and nothing would ever
be worth signing. Fix: give the null contract its own encoding (an `is_null`
flag, third contract-observation feature) and its own sample budget
(`NULL_CONTRACT_FRAC`), rather than letting θ=0 sit at the bottom of a continuous
range. Commit `220a9c3`.

Three 7-agent PHASE1_ONLY arms over a θ sweep (32 envs × 1000 steps, seed 0):

| arm | clean/step at θ=0 | welfare at θ=0 | river at θ=0 |
|---|---|---|---|
| pre-fix baseline | 1.168 | 857 | 90 |
| `is_null` flag, `NULL_CONTRACT_FRAC=0.1` | 0.740 | 66 | 16 |
| flag + `NULL_CONTRACT_FRAC=0.4` | 0.321 | 46 | 10 |

**RETRACTED (2026-08-11) — the numbers.** All three arms were trained and
evaluated in the bugged environment; see the next entry.

**STANDS — the conclusion.** Re-measured in the fixed env, θ=0 cleaning is 0.303
and welfare 47, against ~1.02 and ~2400 contracted. The disagreement point is
genuinely uncontaminated and a contract is worth a ~50× welfare gain. The fix
worked; only its magnitudes were wrong.

Note for any re-analysis: compare arms **at θ=0**, which means the same thing
everywhere, not at θ=max — the ranges differ between arms. At matched θ≈0.5 the
flag alone does not widen the cleaning *gap*; it lowers the θ=0 *level*, and the
level is what matters.

## 2026-08-11 — edge-of-grid cleaning exploit (commit `e2e799e`)

Agents standing at the river's edge, facing the border with one dirt tile
between, could fire the cleaning beam and be credited every step while the tile
was never cleared. Two bugs compounding: a value-scatter that clobbered the hit
record (`.at[].set()` with duplicate indices is order-undefined in JAX), and
double-counted beam slots. Verified before the fix at 25/25 steps credited while
the cell stayed dirt. Fixed with a boolean hit mask built via `.at[].max()`.

Since θ pays *per cleaned cell*, the phantom credit fed straight into transfers
and into the value table every contracting decision is made against.

Re-evaluating the **same** 0.4-arm policies in the fixed env:

- ~35–45% of all measured cleaning credit was phantom (1.5–1.8 → ~1.02 cells/step).
- Welfare under contracts was **understated** by 30–40% (1776 → 2340 at θ=0.2),
  because the unclearable edge cells now actually get cleaned.
- The apparent graded θ-response inside [0.2, 1.0] was **entirely artefact**.
  Real cleaning is flat (~1.02) at every contracted θ. The contract has a
  **threshold** effect at θ=0.2, not a graded one.

**RETRACTED: θ≈0.52 is not the equality target.** Post-fix, parity between
cleaners and harvesters occurs at **θ≈0.87**, and welfare peaks around θ≈0.68. Do
not reuse the 0.52 figure — it appears in earlier notes and is wrong.

**STANDS.** The contract space still spans equality, so the inequality between
specialised cleaners and harvesters is a **protocol** problem, not a
contract-space problem. This is what motivates the bargaining work.

Everything trained before this commit is void. Policies trained *against* the
exploit may also have learned to seek it, so phase 1 needs retraining, not just
re-evaluation.

## 2026-08-11 — why cleaning looks flat in θ

**STANDS.** Dirt spawns at `dirt_spawn_cells=2 × dirtSpawnProbability=0.5` =
exactly 1.0 cells/step. Cleaning throughput cannot exceed the spawn rate, so a
flat ~1.02 clean/step across contracted θ is a **conservation cap**, not an
unresponsive mechanism. Effort is only a ~30% duty cycle, so agents are nowhere
near capacity-limited.

**STANDS.** The system is *critically poised* at the cap, which is why small
differences matter enormously: 0.942 vs 1.018 clean/step gives river stock 130 vs
77 and welfare 575 vs 2764. Any metric reported as "cleaning rate" is therefore
nearly useless on its own — report river stock or welfare alongside it.

## 2026-08-11 — joint bargaining: two degenerate equilibria

Rubinstein alternating offers trained jointly with gameplay (no MOCA phase
split). Two runs, two different degenerate outcomes:

- **run1 (seed 42) — veto dictator.** Agent 3 votes accept with probability
  0.000, everyone else 1.000. Only A3's offers can pass, so every agreement is
  A3's. *Caveat: the contract range for this run was never recorded, so the
  agreed θ=1.000 cannot currently be interpreted.*
- **run3 (seed 55) — random dictator.** *All* agents vote accept with probability
  1.000. Agreement always lands in round 0, and θ is set by whichever agent the
  random rotation happens to open with.

**STANDS — the diagnosis.** Individual rationality is far too slack to discipline
anyone: harvesters take ~530 under a contract against ~114 under the null, so
accepting almost anything genuinely is optimal. Compounding it, non-pivotal
voters receive no gradient — their vote does not change the outcome, so nothing
teaches them to refuse. Alternating offers cannot bite until the disagreement
point is tight enough that rejection is credible.

## 2026-08-11 — entropy collapse from a negative learning rate (commit `62bac84`)

**STANDS.** `linear_schedule` divided the update count by `NUM_UPDATES_PHASE1`
while joint training actually ran `NUM_UPDATES`. Under a shortened budget (~4e7),
the last ~10% of updates therefore ran at a **negative** learning rate — gradient
*ascent* on the PPO loss — and policy entropy collapsed to 0 within a few
updates. Symptom: welfare and `waste_cleared` drop to zero in a single step and
agents spam one action.

Reference example preserved at `runs/rubensteinV1/run2_step300_weird`. Policy
entropy is now logged, which makes this visible in wandb rather than only at the
end.

## 2026-08-12 — three-arm comparison at the wrong contract ranges

Ran `compare_arms.py` over bargaining (run3, seed 55) against the two MOCA
one-shot baseline seeds and a null arm, 20 episodes × 1000 steps.

**RETRACTED — every number.** The run used the default range [0.2, 1.0] for all
three arms. The bargaining arm was trained on [0.2, 2.0] and the MOCA arms on
[0.0, 2.0]. A mismatched range rescales θ through both the contract observation
the policy reads and the unsquash of the proposal it emits, so all arms were
replayed as mechanisms that were never trained. For scale: a 4-episode smoke test
at the correct range put the bargaining arm's agreed θ at **1.585** against the
**0.789** the invalid run reported.

The retracted figures, recorded only so they are recognisable if they resurface:
bargain 2764±145, moca42 2327±832, moca44 2386±973, null 575±283.

**Consequence, not a finding.** This is what motivated run-config sidecars
(`save_run_config`, written at run start beside the checkpoints) and
[../runs/INDEX.md](../runs/INDEX.md). The range now travels with the weights.

**Not yet re-run.** The corrected command is in [bargaining.md](bargaining.md).
The *structural* observations from that session — MOCA's agreement failures,
bargaining's round-0 unanimity, the always-accept voting — are about behaviour
rather than magnitudes and are expected to survive, but they should be re-read
off the corrected run before being relied on.
