# Alternating-offers bargaining over contracts (`algorithms/MOCA/bargain.py`)

Replaces the take-it-or-leave-it contracting stage of Christoffersen et al. with a
Rubinstein/Ståhl alternating-offers game. Background on the contract space itself is
in [contracts.md](contracts.md).

## Why

The paper's stage is one-shot: one fixed proposer, ν=2 sampled voters, and rejection
nulls the contract for the whole episode. The authors name this as a limitation, and
Proposition 4.5 states the consequence — *"all agents except for the proposing agent
are compensated exactly to the point of indifference."*

Prop 4.5 is not the obstacle, it is the **mechanism**. The proposer holds responders
to their continuation value. One-shot, that continuation is "no contract for the rest
of the episode", which in Clean Up is close to worthless, so the proposer takes
everything. Give the responder a credible next move and the same extraction logic
yields an equitable split — **with no fairness axiom added anywhere**. That is the
whole argument, and it is why this stays a bargaining problem rather than becoming an
inequity-aversion term.

The disagreement path also decays *asymmetrically*: harvesters can hold out while
apple stock stands, but their flow payoff collapses faster than the cleaners' as the
river fills. Cleaner bargaining power therefore rises with delay endogenously. The
commons is the disciplining device.

## The protocol

An episode of T steps splits into K = T / `BARGAIN_SEGMENT` segments.

```
round r = 0 .. K−1, while no contract is in force:
    proposer p(r) offers θ_r
    every other agent casts a Bernoulli accept/reject vote
    if #accept ≥ quorum:  θ_r binds for ALL REMAINING segments — done
    else:                 θ = 0 for segment r, continue to round r+1
never agreed → the null contract for the whole episode
```

Rejection costs **one segment**, not the episode. At `BARGAIN_SEGMENT=100` of a
1000-step episode that is roughly a tenth of the episode's welfare — the single lever
on bargaining power.

### Three departures from the reference

- **Rotating proposer, one per round.** The paper warns that with two or more
  proposers "SPEs may be socially suboptimal" (Appendix A) — but that concerns
  *simultaneous competing* proposers. Alternating offers has exactly one per round,
  so the structure that result needs survives.
- **Unanimity among non-proposers by default**, not ν=2. Under majority with 7 agents
  this becomes Baron–Ferejohn: the proposer buys a minimal winning coalition and
  excludes the rest — and here the excluded minority is exactly the cleaners the
  contract exists to compensate. Unanimity is affordable *only* because rejection now
  costs one segment.
- **Counted votes, not a product of probabilities.** The reference's accept
  "probability" is a clipped coordinate of a Gaussian action, so its log-prob is the
  Gaussian's rather than a Bernoulli's, and the product over ν voters conflates "I
  accept" with "the contract passes".

### Randomised first mover

`BARGAIN_ROTATE_START=random` draws the opening proposer once per episode per env.
Rotation alone is not enough: the SPE of this game is agreement in **round 0**, so
with a fixed order agent 0 proposes on every path actually taken and keeps the
first-mover premium permanently, while rotation only ever shapes off-path
continuation values. `fixed` is the ablation that measures the size of that premium.

## Architecture

`BargainingActorCritic` (`networks.py`) — a small MLP (`BARGAIN_HIDDEN=64`) with
three heads: a Gaussian θ proposal, a Bernoulli vote, and a critic. Deliberately
**parallel to** the existing MOCA networks, not a modification of them: every
pre-bargaining checkpoint keeps its exact load and replay path.

`BARGAIN_ACCEPT_BIAS=1.0` puts a positive prior on accept. Unanimity among 6
responders at an unbiased init fires with probability 0.5⁶ = 1.6%, so without it the
contract is almost never in force and the proposal head never sees a signal. A +1.0
logit lifts that to ~15%.

`BARGAIN_ENT_COEF=0.01` is non-zero, unlike the reference's negotiation stage, whose
Gaussian supplies its own exploration through a learned scale. A Bernoulli vote
saturates to always-accept without an entropy term.

### Features (`BARGAIN_FEATURES`)

Handcrafted, `9 + N` dimensions, selected by a mask so the network shape is constant
across tiers and the ablation is a one-value config change. Handcrafted state is
standard in the learned-bargaining literature; the tiers exist because "handcrafted"
is a fair criticism, and they separate the uncontroversial part from the contentious
one.

| tier | adds | rationale |
|---|---|---|
| `protocol` | rounds left, whose turn, standing offer, #rejections, proposer one-hot | the extensive form itself. The SPE of a finite alternating-offers game is Markov in (round, proposer), so this tier alone can in principle represent the equilibrium. |
| `private` *(default)* | own accumulated return, own cleaning | you know your own payoff history |
| `public` | river stock, mean cleaning | **ablation only.** Telling an agent the average contribution is close to handing it the inequality signal the mechanism should discover for itself — so a fair split at `private` cannot be attributed to it. |

## Reward signal and credit assignment

The bargaining policy is trained on the **realised environment return**, not on a
proxy. Each round's reward for agent i is its base reward plus contract transfers
over that round's segment. It is a **semi-MDP**: rounds are decision points, not
fixed-length steps.

- A **rejected** round pays only its own (null-contract) segment.
- The **accepting** round pays its own segment *plus every remaining segment at
  once*, because bargaining ends there and those segments are its consequence.
- Rounds **after** agreement contain no decision and are masked out of the loss —
  training on them would credit actions that could not have mattered.

`round_gae` implements GAE over that structure. **`BARGAIN_GAMMA=1.0`** and should
stay there: Rubinstein's discount factor models impatience, but here impatience is
physically realised — disagreement burns real reward in the environment. Discounting
rounds on top would count the delay cost twice and inflate the proposer's advantage.

## Training mode

`TRAINING_MODE=joint` — one loop, nothing frozen, gameplay and bargaining learning
together for the whole budget. This **drops MOCA's subgame-perfection argument**: the
contract policy chases a moving gameplay policy, and the gameplay policy only ever
sees contracts near what is currently proposed. It exists to get the bargaining game
right before layering the two-phase construction back on top.

Under `joint` the run is tagged `JOINT`, not `MOCA`, in wandb. `--algo MOCA` selects
a directory, nothing more.

```bash
python algorithms/train.py --algo MOCA --env cleanup reward=individual \
  TRAINING_MODE=joint PHASE2_MODE=bargain \
  BARGAIN_SEGMENT=100 BARGAIN_PROPOSER=rotate BARGAIN_ROTATE_START=random \
  BARGAIN_QUORUM=all BARGAIN_FEATURES=private \
  CONTRACT_LOW=0.2 CONTRACT_HIGH=2.0 \
  SEED=55 +ENV_KWARGS.num_agents=7
```

The resolved config is written to `./checkpoints/moca/<stem>.run.yaml` at run start.
**Download it along with the `.pkl` files** — it is what makes the run
interpretable later.

## Replaying and evaluating

The contract range now comes from the sidecar, so these need no range flags:

```bash
# statistics + per-agent voting/proposal pattern, no rendering
python algorithms/MOCA/evaluate_bargain.py \
  --checkpoint 'runs/rubensteinV1/run3_step250/clean_up_seed55_reward_individual_agents7_bargain_seg100_joint_[0-9].pkl' \
  --episodes 20 --num-steps 1000

# against the one-shot MOCA baseline and a null arm
python algorithms/MOCA/compare_arms.py \
  --run "bargain55=runs/rubensteinV1/run3_step250/clean_up_seed55_reward_individual_agents7_bargain_seg100_joint_[0-9].pkl" \
  --run "moca42=runs/moca_baseline/7agents/seed42/clean_up_seed42_reward_individual_agents7_negotiate_nu2_[0-9].pkl" \
  --run "moca44=runs/moca_baseline/7agents/seed44/clean_up_seed44_reward_individual_agents7_negotiate_nu2_[0-9].pkl" \
  --null --episodes 20 --num-steps 1000

# interactive replay; the panel under the per-agent display shows, per round,
# the theta offered, who offered it, and how each agent voted
python viz/interactive_viewer.py \
  --checkpoint 'runs/rubensteinV1/run3_step250/..._joint_[0-9].pkl'
```

For a run with no sidecar the tools print `(fallback)` and warn. Pass
`--range LABEL=LOW:HIGH` (compare_arms) or `--contract-low/--contract-high` and
believe the warning until you have recovered the real bounds from wandb.

## Known failure modes

Both runs so far converged to **degenerate equilibria**, in opposite directions —
a veto dictator (one agent always rejects, so only its offers pass) and a random
dictator (everyone always accepts, so round-0 agreement hands θ to whoever opens).
Details and diagnosis in [findings.md](findings.md).

The shared cause: individual rationality is far too slack for rejection to be
credible, and non-pivotal voters receive no gradient. Alternating offers cannot bite
until the disagreement point is tight enough. Levers, in rough order of directness:
raise `BARGAIN_SEGMENT` so rejection costs more; tighten the range floor; and note
that the phase-1 null-contract conditioning work is what makes the disagreement point
sharp in the first place.

Also watch for the entropy-collapse signature (welfare and `waste_cleared` to zero in
one step, agents spamming one action). The cause — a negative learning rate under a
shortened budget — is fixed in `62bac84`, and policy entropy is now logged so it is
visible in wandb rather than only post mortem.

## Related literature

- **Rubinstein (1982), Ståhl (1972)** — alternating offers; unique SPE under
  discounting.
- **Baron & Ferejohn (1989)** — multilateral bargaining with random recognition;
  minimal winning coalitions. This is what majority quorum degenerates into, and why
  unanimity is the default here.
- **Moulin (1980)** — median-voter strategy-proofness. Because θ is 1-D and V_i(θ) is
  plausibly single-peaked, a median-of-proposals rule would be DSIC. Kept in reserve
  as the manipulation-resistant alternative to a proposer.
- **Nash (1950), Kalai–Smorodinsky (1975)** — axiomatic solutions, available here as
  `SOLVER_DECISION_RULE` arms to compare a *learned* outcome against a *computed* one.
- **Ostrom** — proportional equivalence between contribution and appropriation, the
  motivation for `BARGAIN_PROPOSER=contribution`.

Two ideas were ruled out against the `protocols.py` sandbox and should not be
revisited: **highest-counteroffer-wins** (bidding the ceiling is dominant) and
**highest-acceptance-probability-wins** (approval voting collapses to the majority's
peak, i.e. the harvesters).
