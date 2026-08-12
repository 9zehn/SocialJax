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
    proposer p(r) offers θ_r                                 ← forward pass 1
    every other agent SEES θ_r and casts a Bernoulli vote     ← forward pass 2
    if θ_r = 0:           θ = 0 for segment r ONLY, continue — a formal "pass"
    elif #accept ≥ quorum: θ_r binds for ALL REMAINING segments — done
    else:                 θ = 0 for segment r, continue to round r+1
never agreed → the null contract for the whole episode
```

The θ_r = 0 branch makes the null offer **one segment of disagreement, never a
lock**: whatever the vote, the segment plays uncontracted and negotiation reopens.
Without it, an accepted θ=0 would bind "no transfers" for the rest of the episode —
strictly worse than plain disagreement, which at least keeps renegotiation open.
Votes on a null offer are outcome-free and are masked out of the vote loss. Null
offers arise from null probes (`BARGAIN_PROBE_NULL_FRAC`), and — at
`CONTRACT_LOW=0` — from proposers themselves, since the clipped unsquash puts an
atom of proposal mass at exactly the lower bound, so "propose 0" is a move the
policy can genuinely play.

Rejection costs **one segment**, not the episode. At `BARGAIN_SEGMENT=100` of a
1000-step episode that is roughly a tenth of the episode's welfare — the single lever
on bargaining power.

### Two passes per round, not one

A round is two forward passes over the same network, sharing parameters and
distinguished by a 0/1 "offer live" flag in the state. It has to be. Until
`2026-08-12` both heads were driven by a single pass built *before* anyone moved, so
the vote conditioned on (round, whose turn, last round's already-dead offer, own
stats) and **not on θ_r**. A reservation value — "reject anything below what I could
get by waiting" — was not merely unlearned, it was outside the policy class. The
only expressible vote strategies were proposer-conditioned and constant, which is
exactly the pair of degenerate equilibria both early runs found.

Each agent's **decision-point state** is therefore different: the proposer decided
before the offer existed, the responders after it did. Training records the row each
agent actually acted from — pass 1 for the proposer, pass 2 for everyone else — and
`bargain_loss` recomputes log-probs from that row, so the PPO ratio stays exactly
on-policy. The critic value comes from the same pass for the same reason: a θ-blind
baseline cannot credit a rejection against the size of the offer that was refused.

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

`BARGAIN_PROPOSER=holdout` is the arm that sharpens the rotation further: the next
proposer is drawn uniformly among **last round's rejecters** (random recognition
when there are none — round 0, or a passed null offer). In Rubinstein's two-player
game the refuser *is* the next proposer; rotation only approximates that with seven
players, leaving rejection's payoff two coordinated moves away (reject, then hope
the rotation reaches you). Holdout collapses it to one: reject and you may hold the
pen. Rejecting purely to farm proposal power is priced by the segment the rejection
burns.

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

Under `BARGAIN_VOTE_ADVANTAGE=counterfactual` it grows two more: `lock_value` and
`cont_value`, off the same trunk. They are built **only** in that mode and appended
after the critic, so with the default the parameter tree and its initialisation are
bit-for-bit what they were before the heads were written. Which kind a checkpoint is
gets read off the weights (`bargain.params_have_aux_heads`), never off a flag.

`BARGAIN_ACCEPT_BIAS` puts a positive prior on accept. Unanimity among 6 responders
at an unbiased init fires with probability 0.5⁶ = 1.6%, so without it the contract is
almost never in force and the proposal head never sees a signal; +1.0 lifts that to
~15%. The base yaml now runs it at **0.5** as an ablation: probes supply offers to
vote on whatever the proposers do, and the null-pass rule means a failed round costs
one segment rather than the episode, so the cold-start problem this was for is mostly
handled elsewhere — while the prior itself points at exactly the always-accept
failure under investigation. The code default stays 1.0.

`BARGAIN_ENT_COEF=0.01` is non-zero, unlike the reference's negotiation stage, whose
Gaussian supplies its own exploration through a learned scale. A Bernoulli vote
saturates to always-accept without an entropy term.

`BARGAIN_VOTE_EPS=0.05` is a harder guarantee than the entropy term, which did not
hold: at **rollout** time the accept probability is clipped into `[ε, 1−ε]` before
sampling, annealed linearly down to `BARGAIN_VOTE_EPS_END` over training. Saturation
is self-sealing — an agent that always accepts never observes what refusing would
have bought it — and both prior runs saturated anyway, in opposite directions. The
floor holds both branches open symmetrically, unlike `BARGAIN_ACCEPT_BIAS`, which is
only an initialisation and points one way. The stored old log-prob is the **floored**
one, so PPO's ratio is a proper importance weight against the distribution actually
sampled from; the numerator stays the unfloored policy, or the clip would kill the
gradient of exactly the saturated agents the floor exists to rescue.
`evaluate_bargain` and the viewer replay at ε=0: it is a training device, not part of
the mechanism.

`BARGAIN_VOTE_EPS_END=0.02` (base yaml; the code default is 0, the old behaviour) is
the floor that **remains** at the end of training, and it exists because annealing to
0 was watched failing: in the fixesV1 run, offered θ rose to ~1.9 while acceptance
was still uncertain, then slid back toward ~1.3 — starting at almost exactly the
update where the anneal extinguished the last sampled rejections. Rejection is a
policing strategy; it pays only when someone lowballs, so it is only maintained while
it is occasionally exercised. The persistent floor keeps the threat alive for as long
as proposers are still learning. It shapes training rollouts only — the checkpointed
weights are the un-floored policy either way.

`BARGAIN_PROBE_FRAC=0.1` (base yaml; code default 0 = off) replaces the proposer's
offer, in that fraction of rounds, with a **scripted probe** drawn uniformly over the
whole contract range. Votes are cast and trained on it as on any offer — a probe that
passes binds, which is what makes refusing it worth learning — while the proposal
head is masked out of the round (it did not choose the offer), and the
`theta_offered` metric excludes probes so it keeps reporting the policy's own asking
price. The reason is calibration: a vote threshold only stays sharp on offers it
keeps seeing, and converged proposers cluster (the seed-42 eval made almost no offers
between 0.5 and 1.7, so any threshold there had gone stale — the opening the
lowballers walked through). Probes keep testing the whole range: lowballs the
cleaners must refuse, and — once the ceiling gives them a reason — exorbitant offers
the harvesters must refuse.

`BARGAIN_PROBE_NULL_FRAC=0.2` makes that fraction *of the probes* the null contract
itself. Under the θ=0 protocol rule above these never lock, so they are **forced-null
exposure through the probe channel**: even once every episode agrees in round 0,
gameplay keeps meeting θ=0 segments, which is what keeps the disagreement point —
and with it the meaning of rejection — behaviourally real rather than a stale value
estimate. (Their votes are outcome-free and excluded from the vote loss.)

### Features (`BARGAIN_FEATURES`)

Handcrafted, `12 + 2N` dimensions, selected by a mask so the network shape is
constant across tiers and the ablation is a one-value config change. Handcrafted
state is standard in the learned-bargaining literature; the tiers exist because
"handcrafted" is a fair criticism, and they separate the uncontroversial part from
the contentious one.

| tier | adds | rationale |
|---|---|---|
| `protocol` | rounds left, whose turn, **the offer on the table + a 0/1 live flag**, the standing rejected offer, #rejections, proposer one-hot, **last round's per-agent votes and accept count** | the extensive form itself. The SPE of a finite alternating-offers game is Markov in (round, proposer, offer), so this tier alone can in principle represent the equilibrium — but only with the offer in it. |
| `private` *(default)* | own accumulated return, own cleaning | you know your own payoff history |
| `public` | river stock, mean cleaning | **ablation only.** Telling an agent the average contribution is close to handing it the inequality signal the mechanism should discover for itself — so a fair split at `private` cannot be attributed to it. |

Two details in the `protocol` tier earn their place. The live-offer flag is separate
from the live-offer value so that "nothing has been offered yet" is distinguishable
from "an offer worth 0" — θ=0 normalises to a real number, so without the flag the
proposal pass and a midpoint offer are the same vector. And last round's votes are
the concession signal: "5 of 6 accepted" and "0 of 6" call for very different next
offers, and *which* agent refused tells the proposer whom it has to buy. The
proposer's own slot reads 0, since the quorum does not count its vote.

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

Advantages are standardised over **active rounds only** (`masked_standardise`). The
K×E block is mostly structural zeros — agreement in round 0 is the SPE, and it is
what the runs do — so normalising over the whole block would drag the mean toward 0
and inflate the scale in proportion to how *quickly* the agents agreed, rather than
how good the decision was.

### Counterfactual credit for the vote (`BARGAIN_VOTE_ADVANTAGE`)

The round advantage above answers the wrong question for a *vote*. It is dominated by
the lump-sum agreement reward — the locking round books every remaining segment at
once — so it says "this round went well", not "**my vote** made it go well". Every
responder in a round collects the same signal, including the ones whose vote changed
nothing.

Worse, the noise has a sign. The ε floor samples rejections uniformly, but offers are
not uniform: most rounds carry an offer good enough that everyone else accepts, so a
floored rejection usually lands on a *good* offer, delays it by a segment, and is
punished. The exploration meant to teach *when* refusing pays instead teaches that
refusing costs. That is the [0, 3] run in
[thesis_findings.md](thesis_findings.md): offer visible, exploration persistent,
probes running, and the acceptance curves still flattened to a 0.021 spread with
every slope the right sign and none of them steep enough to discipline anyone.

`BARGAIN_VOTE_ADVANTAGE=counterfactual` replaces it with what the vote actually
changed. The binary action and the counting quorum make the counterfactual exact
rather than estimated:

```
others_accept = n_accept − (my own counted vote)
pivotal_i     = (others_accept == quorum − 1)     # my vote decides, and only then
direction_i   = +1 if I accepted, −1 if I rejected
A_i           = pivotal_i · direction_i · (lock_value_i − cont_value_i)
```

`lock_value` and `cont_value` are two extra heads on the same trunk as the vote,
estimating agent i's remaining return if this offer binds now versus if the round
plays a null segment and bargaining reopens. Each is regressed on the return-to-go of
the rounds where **that branch was realised** — `lock` on rounds that locked, `cont`
on the rest (null-offer rounds included; they are pure continuation samples) — under
`BARGAIN_VF_COEF`, with `stop_gradient` between them and the policy. This is COMA's
counterfactual baseline, specialised: with a binary action and a known pivot rule the
marginalisation over the action space collapses to one comparison.

The advantage is **scaled to unit RMS, never centred** (`bargain.masked_scale`).
A GAE advantage needs centring because its baseline is an estimate; the
counterfactual advantage is already measured against its baseline — the other
branch — so its batch mean is signal. The first cf run demonstrated the failure
centring causes: harvesters held `lock − cont < 0` at every θ (correct beliefs)
while their acceptance *level* sat untrained at ~0.9 with only a slope forming —
centring a one-signed advantage strips exactly the level and leaves the slope,
and PPO's clip truncates the rare large reject-reinforcements centring creates.

Three consequences worth stating plainly:

- **Non-pivotal votes are masked out of the policy gradient entirely**, not
  down-weighted. Their true counterfactual is zero, so any credit they carried was
  noise fitted to what other agents happened to do. The entropy term keeps the wider
  mask: a non-pivotal vote is still a vote the agent will cast again, and letting it
  saturate for want of regularisation is how the always-accept equilibrium formed.
- **Accepting a lowball now yields a negative advantage**, because `lock < cont` at a
  bad offer, so the accept probability falls. This is the gradient the correlational
  advantage could not express in any weight configuration.
- **The branch values are only as good as their generalisation across branches.** At
  a round that locked, the continuation value is a counterfactual never observed
  there. What makes it estimable is that the *same* (θ, round) region gets visited in
  both branches — which is exactly what the ε floor and the probes supply. The three
  mechanisms are load-bearing together, and turning off the floor or the probes
  degrades this too.

The proposal head keeps the GAE advantage unchanged: a proposal has no pivotality and
no branch structure. `round_gae` and the main critic are untouched.

`BARGAIN_VOTE_ADVANTAGE=gae` is the code default and is exactly the pre-2026-08-12
behaviour, down to the parameter tree — the branch heads are not built at all, so
checkpoints from either mode load, and `params_have_aux_heads` tells the eval tools
which they are holding.

**Expected experimental signature** (not testable in CI): the θ-sweep curves in
`probe_votes.py` steepen into genuine 0.5 crossings — cleaners refusing low, and at
the [0, 3] range harvesters refusing high. `probe_votes` also prints the believed
lock-continue gap beside p(accept), which separates *wrong beliefs* from *right
beliefs acted on wrongly*.

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
  BARGAIN_VOTE_EPS_END=0.02 BARGAIN_PROBE_FRAC=0.1 BARGAIN_PROBE_NULL_FRAC=0.2 \
  BARGAIN_VOTE_ADVANTAGE=counterfactual BARGAIN_ACCEPT_BIAS=0.5 \
  CONTRACT_LOW=0.0 CONTRACT_HIGH=3.0 \
  SEED=55 +ENV_KWARGS.num_agents=7
```

Everything on the third and fourth lines is the base yaml's current default, spelled
out so the run's wandb config shows what it was. The code defaults are the *old*
behaviour in every case (`gae`, no floor at the end, no probes, accept bias 1.0), so
the yaml is what carries the experiment line and a config built in code — a test, a
golden arm — is unaffected by it.

The resolved config is written to `./checkpoints/moca/<stem>.run.yaml` at run start.
**Download it along with the `.pkl` files** — it is what makes the run
interpretable later.

## Replaying and evaluating

The protocol and the contract range both come from the sidecar now, so these need no
flags (the paths below are illustrative — the `rubensteinV1` runs are version 1 and
will be refused; see *Checkpoint compatibility*):

```bash
# statistics, per-agent voting/proposal pattern, and accept rate binned by the theta
# on the table -- the table that says whether voting is offer-conditioned at all
python algorithms/MOCA/evaluate_bargain.py \
  --checkpoint 'runs/<run>/clean_up_seed55_..._bargain_seg100_joint_[0-9].pkl' \
  --episodes 20 --num-steps 1000

# against the one-shot MOCA baseline and a null arm
python algorithms/MOCA/compare_arms.py \
  --run "bargain55=runs/<run>/clean_up_seed55_..._bargain_seg100_joint_[0-9].pkl" \
  --run "moca42=runs/moca_baseline/7agents/seed42/clean_up_seed42_reward_individual_agents7_negotiate_nu2_[0-9].pkl" \
  --run "moca44=runs/moca_baseline/7agents/seed44/clean_up_seed44_reward_individual_agents7_negotiate_nu2_[0-9].pkl" \
  --null --episodes 20 --num-steps 1000

# interactive replay; the panel under the per-agent display shows, per round,
# the theta offered, who offered it, and how each agent voted
python viz/interactive_viewer.py \
  --checkpoint 'runs/<run>/..._joint_[0-9].pkl'

# sweep theta through the trained vote head directly -- no rollout, no sampling
# noise. Exact at round 0, where the bargaining state is fully determined by
# (theta, proposer). This is how to locate a reservation threshold that the
# on-policy offer distribution never tests (converged proposers cluster, so
# evaluate_bargain's middle theta bins are often nearly empty).
python algorithms/MOCA/probe_votes.py \
  --checkpoint 'runs/<run>/..._joint_[0-9].pkl'
```

All three replay the **two-pass** round, at ε=0 on the vote floor. They have to: a
single-pass replay would hand the vote head an empty offer slot it never saw in
training, which is not the same policy.

For a run with no sidecar the tools print `(fallback)` and warn. Pass
`--range LABEL=LOW:HIGH` (compare_arms) or `--contract-low/--contract-high` and
believe the warning until you have recovered the real bounds from wandb.

## Known failure modes

Both runs before `2026-08-12` converged to **degenerate equilibria**, in opposite
directions — a veto dictator (one agent always rejects, so only its offers pass) and
a random dictator (everyone always accepts, so round-0 agreement hands θ to whoever
opens). Details in [findings.md](findings.md).

**The first cause was structural, and is fixed.** The vote head never saw the offer
it was voting on, so those two outcomes — proposer-conditioned and constant — were
the *only* strategies the policy class contained. Nothing about the environment or
the incentives was being measured. Every number from a bargaining run trained before
the two-pass round is void for anything about voting behaviour.

**The second cause is behavioural, and the first post-fix run narrowed it.** The
seed-42 evaluation (2026-08-12, [thesis_findings.md](thesis_findings.md)) showed
offer-conditioned voting for the first time — p(accept) correlated +0.95 with θ,
carried by the two heaviest cleaners — but the thresholds were **soft**: a θ=0.2
lowball still passed 53% of the time, so lowballing retained positive expected
value. Two causes were identified and are now countered in code: the exploration
floor annealed to 0 exactly as rejection was becoming profitable
(`BARGAIN_VOTE_EPS_END` keeps it alive), and thresholds go stale on offers the
converged proposers no longer make (`BARGAIN_PROBE_FRAC` keeps testing the whole
range). A third adjustment is the range ceiling: cleaner proposals sat *at* the old
ceiling of 2.0 — a corner solution — and even there cleaners barely reached parity,
so the ceiling, not bargaining, was setting the split. The ceiling should be high
enough that the agreed θ settles in the interior (3.0 for the current line), which
also gives harvesters a genuine rejection region of their own at the top of the
range. Held in reserve if thresholds stay soft: forced-null exposure, and the phase-1
reconstruction, which sharpens the disagreement point by construction.

**The third cause is credit assignment, and it is what the [0, 3] run isolated.**
With the offer visible, the floor persistent and probes running, the acceptance
curves *still* flattened — spread 0.021, correlation with θ a healthy +0.97, slopes
5–10× too small to make lowballing unprofitable. Since the data was there (13
sub-0.5 offers in the eval alone), neither exploration nor representation was the
binding constraint: the vote was being credited with the round's shared outcome
rather than with its own effect, and the ε floor's rejections landed mostly on good
offers, so correlational credit actively taught that refusing costs. That is what
`BARGAIN_VOTE_ADVANTAGE=counterfactual` addresses — see *Counterfactual credit for
the vote* above.

Distinguishing these is now a table rather than an argument: the *voting vs the
offer* block in `evaluate_bargain.py` reports accept rate and mean p(accept) binned
by the θ on the table, `probe_votes.py` sweeps the vote head directly, and under
counterfactual credit it also prints the believed lock-continue gap. A flat p(accept)
row with a **correctly signed gap** is a policy that knows better and does not act on
it — a credit or optimisation problem. A flat row with a flat gap is an agent that
does not believe the offer matters, which points back at the environment and the
disagreement point.

Also watch for the entropy-collapse signature (welfare and `waste_cleared` to zero in
one step, agents spamming one action). The cause — a negative learning rate under a
shortened budget — is fixed in `62bac84`, and policy entropy is now logged so it is
visible in wandb rather than only post mortem. `joint/health/vote_eps` sits alongside
it: while the floor is still non-zero, part of the accept rate is the floor rather
than the policy.

## Checkpoint compatibility

The feature layout is versioned (`bargain.FEATURE_VERSION`, currently **2**) and
written into each run's `.run.yaml` sidecar as `BARGAIN_FEATURE_VERSION`. The
evaluator, `compare_arms` and the viewer all refuse a mismatch — by recorded version,
or failing that by parameter shape — with an explicit error rather than replaying a
checkpoint as a mechanism it was never trained on.

Version 1 runs (everything in `runs/rubensteinV1/`) **cannot be loaded by this code**:
the feature vector went from `9 + N` to `12 + 2N`, so the first layer's weights are
the wrong shape. To replay one, check out the commit recorded in its sidecar. Nothing
is lost, but nothing is silently reinterpreted either.

The counterfactual branch heads are **not** a version bump — the feature layout and
the protocol are unchanged, only the loss and the parameter tree. A
counterfactual-trained checkpoint simply carries four extra parameter groups, and
every replay tool infers their presence from the weights themselves
(`params_have_aux_heads`) before building the network. So both kinds of checkpoint
replay with no flags, in either direction, and there is nothing for a user to get
wrong.

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
