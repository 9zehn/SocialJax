# Formal contracting (`algorithms/MOCA/`)

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

## Three environments, three contract spaces

Every space is one scalar over {0} ∪ [low, high] with the same observation encoding,
so an arm means the same thing wherever it is run. They differ only in which
observable outcome the transfer reads and where the money goes.

| env | θ buys | range | env signal |
|---|---|---|---|
| `clean_up` | payment per waste cell cleaned, funded evenly by the others | [0, 0.2] | `cleaned_by_agent` |
| `harvest_common_open` | fine on eating an apple in a low-density patch, split evenly among the others | [0, 10] | `low_density_eaten` |
| `coin_game` | payment per coin taken of another agent's colour, paid to its owner | [0, 2] | `stolen_by_agent`, `stolen_from_agent` |

The first two are the authors' verbatim (`CleanupContract`,
`HarvestFeaturemodLocalContract`). **The third is not theirs** — their release covers
Cleanup, Harvest and a self-driving domain only, so the Coin Game space is built to
their design rules rather than transcribed. Report it as constructed.

Which space goes with which environment, plus the per-env behaviour metrics, lives in
[../algorithms/MOCA/envs.py](../algorithms/MOCA/envs.py). The training loop is
env-agnostic; adding an environment is an entry there and a contract class in
`contracts.py`.

### Clean Up: subsidise the under-provided good

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

### Harvest: tax the over-used resource

The sign is the **opposite** of Clean Up's, and that is the point of running both.
Clean Up prices provision; Harvest prices appropriation. They are the two halves of
Ostrom's pair, and a mechanism that fixes one need not fix the other.

```
pay_i      = θ · d_i                          # d_i ∈ {0,1}: ate in a thin patch
receive_i  = θ · (Σ_j d_j − d_i) / (N − 1)    # your share of everyone's fines
transfer_i = receive_i − pay_i
```

"Thin patch" is the reference's predicate: fewer than 4 apples with `j² + k² ≤ 5` of
the agent — a 21-cell neighbourhood, because their code compares squared distance
against the *radius* rather than its square. Kept literal rather than corrected: the
threshold of 4 is calibrated against that neighbourhood, and widening one without the
other redefines which harvesting is charged for. Both are config
(`low_density_radius`, `low_density_threshold`).

Why local density rather than a flat harvest tax: apples regrow at a rate that
depends on how many remain nearby and stop regrowing once a patch is stripped, so
eating the last apples of a patch is the one act with a lasting external cost. A flat
tax would price that the same as eating from a full patch and suppress the behaviour
the commons is *for*.

θ is a **flat charge per qualifying step**, not per apple — as in the reference, and
identical here since an agent moves onto at most one cell per step.

The range is 50× Clean Up's because it prices a different thing against a different
base: one eating event, at most once per agent per step, against a unit apple. **θ = 1
is the analytic floor** — below it, eating the last apple of a patch still nets a
profit and the contract cannot bind.

### Coin Game: pay the agent you stole from

```
pay_i      = θ · (coins i took that were not i's colour)
receive_i  = θ · (coins of i's colour that others took)
transfer_i = receive_i − pay_i
```

Zero-sum because every taken coin appears once on each side. Follows
`SelfdriveContractDistprop` rather than the two grid-world spaces in paying the harmed
party: a taken coin has exactly one owner. At the 2 agents this environment supports
the two conventions are the same transfer; paying the victim is the one that keeps
meaning what it means if the env is ever widened.

The range is set by the payoff matrix (+1 for any coin, −2 to its owner), not tuned:
**θ = 1** is where stealing stops paying, **θ = 2** is where the owner is made whole.
Above 2 the contract would pay agents to be robbed.

### Calibrate the reward scale or every θ is N× too weak

All three environments default their primary reward to `num_agents`, so that
individual- and shared-reward arms carry equal reward mass. Every contract range is
quoted against a **unit** reward. The configs set `apple_reward: 1.0` (Clean Up,
Harvest) and `coin_reward: 1.0` (Coin Game) explicitly; `make_train` warns when it is
missing, because nothing else about the run looks wrong.

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

## Two ways to train it (`TRAINING_MODE`)

The comparison the experiments turn on. Same environment, same contract space, same
protocol, same range — differing only in whether the two-phase construction is used.

| mode | what it is |
|---|---|
| `two_phase` | **MOCA**, Algorithm 1 below. |
| `combined` | **Single-stage contracting.** A contract is negotiated afresh at the start of *every* episode and gameplay learns under it. Nothing frozen, no P(Θ). The reference's `SeparateContractCombinedStage`. |
| `joint` | Alternating-offers bargaining, this repo's extension — see [bargaining.md](bargaining.md). Runs on all three environments. |

`combined` requires `PHASE2_MODE=negotiate`: `solver` scores contracts with a critic
that only means anything once gameplay is frozen, and `reinforce` is a bandit over a
fixed subgame. Neither is single-stage contracting.

What `combined` gives up — the reason MOCA exists, and what the gap between the two
arms measures:

* gameplay meets only contracts the current proposer likes, so V_i(s₀, θ) is
  estimated on-distribution rather than across the space;
* the proposer is scored against a gameplay policy that is still moving;
* the null contract appears only when the negotiation *fails*, so the disagreement
  point every acceptance decision is implicitly measured against may be barely
  sampled at all. Watch `combined/contract_null_frac` — under MOCA that is
  `NULL_CONTRACT_FRAC` and fixed by construction; here it is an outcome of the run,
  and a run that drives it to 0 has stopped estimating V_i(s₀, 0).

`TRAINING_MODE` is in the checkpoint name (`two_phase` unmarked, so existing MOCA
filenames are unchanged), so the two arms cannot overwrite each other.

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
4. **θ range raised** past the paper's [0, 0.2] — *Clean Up only*. That range was
   calibrated to upstream's 0.5 dirt/step; under this repo's ecology the negotiated
   contract pinned to the 0.2 ceiling — a boundary solution, meaning the reported θ
   measured the limit of the range rather than what agents wanted. Overpaying
   cleaners makes the harvesters funding it worse off than the null, and they reject,
   so the range only needs to *contain* the optimum. **Harvest keeps the paper's
   [0, 10]**: nothing there has been measured yet, so start where the authors did and
   widen only if θ pins to a boundary.
5. **Extra solver decision rules** (Nash, Kalai–Smorodinsky, egalitarian).
6. **Alternating offers** as a fourth arm.
7. **A Coin Game contract space**, which the authors do not define at all.

## Running it

```bash
# MOCA, per environment (arm=moca is the default)
python algorithms/train.py --algo MOCA --env cleanup
python algorithms/train.py --algo MOCA --env harvest
python algorithms/train.py --algo MOCA --env coins

# the other arms, one flag each
python algorithms/train.py --algo MOCA --env harvest arm=vanilla       # no phase split
python algorithms/train.py --algo MOCA --env harvest arm=renegotiate   # per-segment
python algorithms/train.py --algo MOCA --env harvest arm=episode_lock  # binds the episode
python algorithms/train.py --algo MOCA --env cleanup arm=median        # Clean Up only
```

`--env <stem>` resolves to `algorithms/MOCA/moca_cnn_<stem>.py` and
`config/moca_cnn_<stem>.yaml`. The three entry modules are thin; the algorithm is in
`moca_cnn.py`. `arm=` is a config group (`config/arm/`), last in each per-env defaults
list so it overrides everything above it.

### The welfare ceiling (`--algo JOINT`)

```bash
python algorithms/train.py --algo JOINT --env cleanup
```

Christoffersen et al.'s `joint` baseline (`JointEnv`): **one** network sees every
agent's observation, chooses every agent's action, and is paid their summed reward.
No dilemma remains in that problem, so what it converges to is the best joint
behaviour the environment admits — the number every decentralised arm is read
against. Implementation in [algorithms/JOINT/](algorithms/JOINT/); the network is
[networks.py](algorithms/JOINT/networks.py).

Three things carry over from the reference exactly: the observation is every agent's
egocentric view concatenated **on the channel axis** (their `concatenated_obs`), the
action is the joint action factored into one categorical head per agent, and the
reward is `sum(env_rews.values())` — their "straightforward sum, not average".
`shared_rewards=True` is refused, since the env would then pay each agent the sum
already and summing again would scale welfare by N.

Its PPO settings are copied from `moca_base.yaml` and should be kept in step with
them, or the gap stops measuring the control structure and starts measuring tuning.
Checkpoints go to `./checkpoints/joint/`, keeping them clear of the decentralised
runs, which share the same stem by construction.

**Read the gap, not the level** — and read `joint/equality` next to `joint/welfare`.
The summed objective is indifferent between an even split and one agent taking
everything, so the ceiling can sit at an allocation individual rationality rules out.
A voluntary mechanism cannot be expected to reach a welfare number no agent would
agree to; the runner prints a note when that happens.

Every arm is in the checkpoint stem — `TRAINING_MODE`, `PHASE2_MODE`, and for
bargaining runs the segment length and `BARGAIN_BINDING` — so arms of the same
comparison cannot overwrite one another.

## Evaluating

```bash
# V_i(s_0, theta) over a theta sweep, for a frozen phase-1 policy. Runs on all three
# environments; --env picks the contract space and the behaviour metric.
python algorithms/MOCA/grid_eval.py --checkpoint '<stem>_[0-9].pkl' \
  --env harvest_common_open --num-agents 7 --contract-high 10.0

# like-for-like arm comparison; contract ranges come from each run's sidecar
python algorithms/MOCA/compare_arms.py \
  --run "moca42=runs/moca_baseline/7agents/seed42/..._negotiate_nu2_[0-9].pkl" \
  --null --episodes 20 --num-steps 1000
```

Compare arms **at θ=0**, which means the same thing everywhere — not at θ=max, since
the ranges differ between runs (and now between environments).

**Clean Up only, for now:** `compare_arms.py`, `evaluate.py`, `evaluate_bargain.py`,
`probe_votes.py`, `sweep_theta_utility.py` and the interactive viewer all read
`info["cleaned_by_agent"]` or the bargaining state directly. `grid_eval.py` is the
one that has been generalised, and it is the one the θ-response and
individual-rationality results come from.

## Known issues

- Inequality between specialised cleaners and harvesters is a **protocol** problem,
  not a contract-space problem: the space spans equality (parity at θ≈0.87), but the
  one-shot ultimatum lets the proposer extract to the point of indifference
  (Proposition 4.5).
- Cleaning throughput saturates at the 1.0 cells/step dirt spawn rate, so a flat
  clean/step across θ is a conservation cap, not an unresponsive mechanism. Report
  river stock or welfare alongside it.
