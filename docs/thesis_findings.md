# Thesis findings — distilled for the write-up

Compiled 2026-08-12 (by Fable, not Opus — the CLAUDE.md disclaimer about .md files
applies here too, but every externally-checkable claim below carries a verification
label). Purpose: the short list of findings, design arguments, and cautions that the
dissertation write-up will actually need, separated from the day-to-day log in
[findings.md](findings.md).

Labels: **[VERIFIED]** checked against the primary source on the date given.
**[OURS]** our own design argument or measurement — sound, but ours to defend.
**[PENDING]** plausible but not yet checked; do not quote until verified.

---

## A. Environment-level results that survived the `e2e799e` fix

All from [findings.md](findings.md); re-measured after the edge-of-grid exploit fix
(2026-08-11). Any number from before that commit is void.

- **The exploit itself is a reportable methods point.** ~35–45% of measured cleaning
  was phantom (credit for dirt never cleared); welfare under contracts was
  *understated* 30–40%. Root cause: `.at[idx].set()` with duplicate indices is
  order-undefined in JAX — an instance of how silent numerical semantics can
  invalidate MARL results. **[OURS]**
- **Null-contract conditioning collapses the disagreement point.** With an is_null
  flag and NULL_CONTRACT_FRAC=0.4, θ=0 play gives cleaning 0.303 and welfare ~47,
  versus ~1.02 and ~2400 under a contract — a contract is worth roughly a 50×
  welfare gain and is individually rational for everyone. Without this
  conditioning, agents cooperate at θ=0 and nothing would sign. **[OURS]**
- **The contract's θ-response is a threshold, not a gradient.** Real cleaning is
  flat (~1.02 cells/step) at every contracted θ; the apparent graded response
  inside [0.2, 1.0] was entirely the exploit artifact. Return parity between
  cleaners and harvesters occurs near θ≈0.87; welfare peaks near θ≈0.68. (The old
  θ≈0.52 equality figure is retracted.) **[OURS]**
- **Cleaning rate alone is a nearly useless metric.** The system sits critically
  poised at the dirt-spawn conservation cap (1.0 cells/step): 0.94 vs 1.02
  clean/step is the difference between welfare 575 and 2764. Always report river
  stock and welfare alongside cleaning rate. **[OURS]**

## B. Mechanism-design contributions over Christoffersen et al. (arXiv:2208.10469)

Quotes verified against the paper (ar5iv HTML) on 2026-08-12.

1. **Counted Bernoulli votes fix a real mis-specification.** The reference gates
   contract acceptance by a single random draw against a *product* of accept
   probabilities: it samples ν non-proposing agents and adopts the contract
   "if rand() < ∏ⱼ πⱼ(i,θ)". **[VERIFIED]** So no individual accept/reject event
   ever occurs as an action with its own log-probability — "I accept" is conflated
   with "the contract passes", and per-voter credit assignment is impossible by
   construction. Our redesign gives every responder an explicit Bernoulli vote and
   passes the contract on a counted quorum. *(The further claim that their accept
   probability is implemented as a clipped coordinate of a Gaussian action comes
   from their released code, not the paper — **[PENDING]** check the code before
   quoting that detail.)*
2. **The one-shot ultimatum is a named limitation of the reference, and Prop 4.5
   is the lever.** The paper: formal contracting "does not have the dynamic
   structure of a negotiation and lets a proposing agent make a take-it-or-leave-it
   offer" **[VERIFIED]**; and from the proof sketch of Proposition 4.5: "All agents
   except for the proposing agent are compensated exactly to the point of
   indifference; the proposing agent maximizes its reward, under the constraint of
   agents accepting the contract." **[VERIFIED]** Our core argument: this
   extraction logic is not an obstacle but the mechanism itself — the proposer
   holds responders to their continuation value. One-shot, that continuation is
   "no contract for the rest of the episode" (≈ worthless in Clean Up), so the
   proposer takes everything. Make rejection cost one segment instead of the
   episode and the *same* logic prices the responder's credible next move —
   an equitable split with no fairness axiom added anywhere. **[OURS]**
3. **The paper's multiple-proposer warning does not apply to alternating offers.**
   It warns "if two or more agents may propose in a game, SPEs may be socially
   suboptimal" **[VERIFIED]** — but that concerns simultaneous competing
   proposers; alternating offers has exactly one proposer per round. **[OURS]**
4. **Randomised first mover is load-bearing.** The SPE of finite alternating
   offers is agreement in round 0, so with a fixed order one agent proposes on
   every path actually taken and keeps the first-mover premium permanently;
   rotation alone only shapes off-path continuations. Drawing the opening
   proposer once per episode removes this; `fixed` is the ablation that measures
   the premium. **[OURS]**
5. **Unanimity over ν=2 or majority.** Under majority with 7 agents the game
   becomes Baron–Ferejohn: the proposer buys a minimal winning coalition and
   excludes the rest — and the excluded minority is exactly the cleaners the
   contract exists to compensate. Unanimity is affordable only because rejection
   now costs one segment. **[OURS]**
6. **γ=1 across bargaining rounds, deliberately.** Rubinstein's discount factor
   models impatience, but here impatience is physically realised — disagreement
   burns real reward in the environment. Discounting rounds on top would count
   the delay cost twice and inflate the proposer's advantage. **[OURS]**
7. **Credit over rounds, not steps.** A rejected round is credited with its own
   null segment; the accepting round is credited with all remaining segments at
   once (they are its consequence); rounds after agreement contain no decision
   and are masked out of the loss. **[OURS]**

## C. Empirical findings from the bargaining runs (all pre-fix — behavioural, not final)

The vote-blindness defect (section D) means these runs could not have expressed
offer-conditioned voting; read them as behaviour under that constraint. Numbers are
from the corrected-range evaluation of run3 (seed 55, range [0.2, 2.0]).

- **Bargaining beats the one-shot MOCA baseline where the protocol differs, not
  where it doesn't.** Welfare 2764±145 vs 2327±832 / 2386±973 (moca seeds 42/44);
  the gap comes almost entirely from agreement reliability — 20/20 episodes agreed
  with 0 uncontracted steps, vs 0.85–0.90 agreement and 100–150 null steps for
  one-shot. Per-episode, MOCA's failures are catastrophic (welfare 128–927), not
  marginal. Clean/step under contract is near-identical across arms (~1.02, the
  cap). **[OURS]**
- **At the θ levels actually agreed (θ̄≈1.53), contracts still under-compensate
  cleaning**: cleaner:harvester return ratio 0.890. Cleaners propose the ceiling
  (2.0) every time they hold the pen; harvesters propose ~1.37 and carry 3× more
  agreements (15 vs 5). The surplus a credible rejection threat should claw back
  is visible in the data. **[OURS]**
- **Two degenerate equilibria, in opposite directions, across seeds:**
  - *Veto dictator (seed 42, θ ceiling 1.0):* one cleaner voted accept with
    probability ~0 — only its own proposals (always the ceiling, 1.0) could pass.
    Strikingly, it also **withheld cleaning until the rotation reached its
    proposal turn**. This is strategic holdout expressed through gameplay: the
    agent coupled its environment behaviour to the bargaining state, raising the
    cost of delaying agreement with it. Unwanted as an equilibrium, but direct
    evidence that (a) gameplay policies condition on contract state and (b)
    endogenous bargaining power through the commons is a real channel — the
    thesis's "commons as disciplining device" hypothesis showing up in an
    unintended place. **[OURS]**
  - *Random dictator (seed 55, ceiling 2.0):* all agents vote accept with
    probability ~1; agreement always in round 0; θ set by whoever the random
    rotation opens with (hence agreed θ spread 0.34–2.0 with sd 0.59). **[OURS]**
- **Equilibrium multiplicity across seeds is expected, not anomalous.** Bargaining
  games lack the structural properties standard no-regret convergence results
  need, and which equilibrium learning dynamics select depends on initial
  conditions (cf. arXiv:2507.03150). Two seeds landing on different degenerate
  equilibria is the textbook signature. **[OURS, literature-backed]**

### Post-fix run (fable_fixesV1, contract visible at negotiation) — 2026-08-12

From wandb curves (~275 updates), pending proper evaluation. **[OURS, provisional]**

- **Bargaining discipline is transient under naive joint training.** While
  acceptance was still uncertain (accept_count ~4.3–5.5, in_force_rate < 0.9),
  offered and agreed θ *rose* to ~1.9 — proposers conceded to secure unanimity.
  Once acceptance saturated (~update 150: accept_count → ~6, agreement → round 0),
  θ slid steadily back to ~1.2–1.4 and transfer volume fell from ~1750 to ~1300:
  with rejection extinct, proposers walk θ down unpunished — Prop 4.5's
  hold-to-indifference dynamic reasserting itself *through learning dynamics*
  rather than through the protocol. Welfare barely moves (cleaning is pinned at
  the conservation cap; θ stayed above the 0.2 threshold), so the erosion is
  purely distributional — equality dips late.
- **The erosion is out of phase with exploration.** The vote exploration floor
  (vote_eps 0.05 → ~0) annealed on a time schedule; rejection became profitable
  (θ falling) exactly as the exploration that could discover it disappeared.
- **The architecture fix alone did not change the equilibrium.** The post-fix run
  tracks the pre-fix run on every aggregate metric. Offer-conditioned voting is
  now *representable*; the learning dynamics do not *maintain* it. Whether the
  vote head nonetheless learned a reservation threshold below the on-policy offer
  distribution (proposers stopping at ~1.2 could be rational avoidance of a
  learned threshold rather than unconditional acceptance) is checkable offline by
  sweeping θ through the vote head — do this before concluding the vote is flat.

### Post-fix evaluation (seed 42, 40 episodes × 1000 steps) — 2026-08-12

The aggregate-metric pessimism above was wrong in one crucial respect: the
mechanism *did* learn offer-conditioned voting. **[OURS, from evaluate_bargain]**

- **First run to show offer-conditioned voting.** p(accept) correlates **+0.95**
  with θ; under unanimity the per-voter spread compounds: θ≈0.2 offers pass 52.9%,
  θ≈2.0 offers pass 88.2%. Round-0 acceptance 72.5% (was 100% pre-fix); 11/40
  episodes reach round ≥1; agreement still 100% with only 32/1000 steps
  uncontracted — the added toughness is nearly free in delay.
- **Discrimination is carried by exactly the right agents.** A1 (accept 0.690 at
  θ=0.2 → 0.826 at θ≈2) and A4 (0.794 → 0.914) — the two heaviest cleaners. All
  other responders ≈0.998 flat, including cleaner A5 — a hybrid (0.237
  clean/step, best cleaner return) for whom flat acceptance is arguably rational.
  The discrimination boundary tracks true role exposure, not the role label.
- **Case study: episode 2** — cleaner A1 rejected through three rounds and won
  the ceiling (θ=2.0, agreed R3). The 300-step holdout burned ~900 welfare,
  demonstrating both the credible threat and its real cost.
- **The disagreement point now has behavioural teeth**: uncontracted segments
  clean at 0.565/step vs 1.035 contracted — first run where no-contract play is
  visibly different. Holdout genuinely degrades the river.
- **Outcomes are bimodal — never report the mean θ (1.56) alone**: 29 episodes at
  θ=2.0, 9 at θ=0.2, 2 between. Cleaners all propose the ceiling; harvesters
  split into three lowballers (A0/A2/A3 at 0.2) and one high proposer (A6, 1.94).
- **Lowballing still has positive expected value**: 53% passage means lowballer
  A0 out-earns high-proposing harvester A6, 404 vs 385. Soft thresholds are the
  remaining defect — A1/A4 reject a 0.2 offer at only ~25–30% each, versus the
  near-100% their transfer exposure warrants.
- **The offline θ-sweep (probe_votes.py, same day) settles what "soft" means:
  the acceptance curves are shallow RAMPS, not thresholds.** A1 rises ~0.71→0.83
  and A4 ~0.81→0.92 across [0.2, 2.0], monotone, never crossing 0.5; the other
  five responders are flat ≥0.998. So there is no reservation value anywhere in
  the range — proposers stopping at θ≈1.2 during training were not avoiding a
  cliff, and lowballing was simply a favourable lottery. "Emergence of
  offer-conditioned voting without a reservation value" is the precise
  characterisation for the write-up.
- **Lowball episodes are socially expensive**: θ=0.2 episodes average ~2,280
  welfare / 0.76 equality vs ~2,870 / 0.955 for round-0 θ=2.0 episodes, on a
  clean/step difference of only ~0.03 — the critically-poised-system effect.
  Cleaner:harvester ratio 0.909 overall (from 0.890 pre-fix), still short of
  parity.
- **Write-up caution on aggregates**: welfare 2679±275 / equality 0.910 sits
  slightly *below* the pre-fix always-accept eval (2764±145 / 0.930). Partial
  deterrence admits exploitation episodes the always-accept regime accidentally
  avoided (its proposers never learned to lowball) — closer to equilibrium can
  look worse on averages while the mechanism is strictly healthier.
- Provenance: range taken from flag, not sidecar (confirm the run's `.run.yaml`);
  checkpoint stem collides with the pre-fix seed-42 veto-dictator run — keep
  directories separated in runs/notes.yaml.

### The [0, 3] run with floor + probes + null-pass rule (40 episodes) — 2026-08-12

Same protocol plus: persistent vote floor (`BARGAIN_VOTE_EPS_END=0.02`), scripted
probes (10%, of which 20% null), null offers never lock, range [0, 3]. **[OURS]**

- **Parity reached for the first time: cleaner:harvester return ratio 1.011**
  (0.890 → 0.909 → 1.011 across the three runs) at agreed θ̄=2.42. With room
  above the old ceiling, negotiated redistribution fully compensates cleaning —
  no fairness axiom anywhere.
- **First bilateral discipline signal**: harvesters A0/A3/A4 now show a
  *downward* p(accept) slope toward θ=3 (~0.99 → 0.97) while cleaners slope up —
  both rejection regions predicted for a widened range exist, in the right
  agents.
- **The θ-welfare relationship inverted at the top: θ=3 overshoots.** θ=3
  episodes average ~2,500 welfare; interior-θ episodes (0.67–1.9) reach
  2,800–2,870. Cleaning is capped either way (~1.03); the cost is labour
  diversion — at θ=3 a fourth agent (A3, 0.127 clean/step) is pulled toward the
  river to chase transfer income, and foregone harvesting rots. Meanwhile
  equality is best near θ≈2–3 and collapses at low θ (0.57–0.78). **The contract
  space now brackets a genuine welfare/equality trade-off, and the agreed
  θ̄=2.42 sits on the equality side of it.** Role composition responds to θ: A6
  flipped harvester→cleaner between runs.
- **But the vote FLATTENED: spread 0.021 (was 0.054), minimum p(accept) 0.893.**
  Correlation with θ stays +0.97, magnitude shrank. Floor and probes supplied
  the data (13 sub-0.5 offers in the eval alone), so *exploration and
  representation are no longer the constraint — credit assignment is*: the vote
  advantage is correlational, swamped by the lump-sum agreement reward, and
  never isolates "what did MY vote change". Proposers remain a random
  dictatorship over a wider, costlier range: cleaners corner at 3.0 (27 carried),
  harvesters lowball (13 carried), welfare 2555±219 and equality 0.834±0.101
  both below the [0.2, 2] run.
- **New exploit at the null boundary**: exact θ=0 is a no-lock pass, but θ=0+ε
  locks — and locks happened at θ=0.021 and 0.071 (episodes 17, 39; equality
  0.57–0.59). One notch above the pass move buys an episode of uncompensated
  cleaning. The principled fix is a vote that rejects it (the stake is the
  largest in the whole range); the design wart should be named in the write-up.
- Aggregate caution as before: the welfare/equality dip vs run4 is the dictator
  lottery widening with the range, not the mechanism regressing — per-proposer
  outcomes, not means, are the informative view.

### Counterfactual-vote run ([0,3], cf advantage + bias 0.5) — 2026-08-12

**[OURS, from wandb + probe_votes + evaluate_bargain, 40 episodes]**

- **First genuine bargaining dynamics.** Mid-training crisis (updates ~60–90):
  a rejection wave crashed accept_count to 4 and in_force to ~0.3, and offers
  were forced UP from ~1.5 to ~2.4, where acceptance recovered — rejection-driven
  concession, the alternating-offers mechanism visibly functioning for the first
  time. Eval: R0 acceptance 0.60 (was 0.725), 40% of episodes reach R1–R4,
  lowball locks down to 12% pass. Agreement still 100%.
- **But it settled past the welfare peak, and the rent dissipated by entry.**
  θ̄≈2.1–2.4 pulled a FOURTH agent into cleaning (A3/A4/A5/A6; welfare 2167 vs
  ~2500–2650 in interior-θ episodes; welfare peaks at θ∈[1,2] where A6 harvests
  instead). The transfer pool is capped (θ·~1.02 cells/step) so entry splits it
  more ways while foregone harvest shrinks the pie: cleaners earn 300 vs
  harvesters' 322 — ratio 0.934, BELOW parity despite high θ. Classic
  Gordon/Ostrom rent dissipation: high θ doesn't help cleaners as a class once
  occupational entry is free. The agents' own interest should favour interior θ.
- **Beliefs learned, policy didn't follow (the pre-registered failure mode).**
  The lock−continue heads are coherent: harvesters A0/A2 hold NEGATIVE gaps at
  every θ (locking worse than bargaining on) yet vote accept 0.86–0.92 —
  belief/action mismatch 100% of the θ grid; the other five agents' positive
  gaps are consistent with the now-brutal disagreement point (uncontracted
  welfare/step 0.025). Mechanical suspect: `masked_standardise` CENTERS the
  counterfactual advantage, but COMA's advantage is already self-baselined —
  centering a one-signed advantage (policy systematically wrong) strips most of
  the level signal and leaves only the slope, exactly matching the data (slopes
  formed, acceptance level stuck ~0.9, and PPO's clip caps the rare big
  reject-reinforcements that centering creates). Fix: scale by std only.
- **Unanimity verdict:** it is not muting anyone — it grants A0/A2 a full veto
  they are not using. Majority quorum would WEAKEN harvester vetoes (cleaner
  offers pass 4/6 without them). Keep unanimity; fix the mismatch.

### Uncentered-cf arms: rotate vs holdout — 2026-08-12

**[OURS, wandb only — checkpoints LOST (Colab download killed before finishing;
stream checkpoints to Drive next time). ~300 updates each, seed 42, [0,3].]**

Three-way comparison: centered+rotate (previous run) vs uncentered+rotate (arm 1)
vs uncentered+holdout (arm 2).

- **Un-centering did what it was built to do**: vote levels finally moved.
  Arm 1: accept_count ~5.1, agreement round ~3.7, in_force ~0.62 — real
  rejection at last, but miscalibrated into a war of attrition that still ends
  HIGH (θ̄≈2.75) after burning ~4 segments; welfare ~1450. Pivotal rate fell to
  ~0.33 (more rejection ⇒ fewer rounds where a single vote is pivotal ⇒ less cf
  credit — the credit mechanism is partly self-limiting under rotate).
- **Holdout steered the agreement into the welfare-optimal band**: arm 2 agreed
  at **θ̄≈1.55** (the [1,2] interior identified across runs) with agreement
  round ~1.9, in_force ~0.81, welfare ~2150, less over-cleaning (waste_cleared
  ~115 vs 130). Giving rejecters the pen converts rejection into counteroffers:
  harvesters who refuse a 3.0 propose lower and it sticks. First run where the
  negotiated θ lands in the welfare-optimal interior.
- **The welfare gap vs the centered run (~2450) is disagreement deadweight, not
  worse contracts**: ~2 burned segments ≈ −500 welfare at the near-worthless
  disagreement point. The SPE is round-0 agreement AT the interior θ; arm 2's
  theta_offered was still drifting down (~1.7) at cutoff — proposers learning
  to open acceptably. Expect the gap to close with longer training and/or
  BARGAIN_SEGMENT=50 (halves the per-rejection deadweight).
- **Prediction registered for the missing 2×2 cell (centered + holdout)**:
  holdout only fires on rejections, and centered credit is what suppressed
  rejection levels — so centered+holdout should revert toward the
  round-0-acceptance, θ̄≈2.4 outcome with a random-ish proposer. Worth running
  to complete the {centered,uncentered}×{rotate,holdout} square, with this
  prediction on record.

### seg=50 + holdout: the rejection monopoly — 2026-08-13

**[OURS, probe + 40-episode eval, run8_50segments]** Halving the segment turned
recognition-by-holdout into a **pen monopoly**: A5 votes reject with p=0.000 on
*everything*, so every round it responds to fails, holdout hands it the pen, and
it dictates θ=3.000 — proposer in 40/40 agreements (R0:6 by lottery, R1:34), θ̄ =
2.9996 ± 0.003. This is exactly the strategic risk the mechanism priced by the
burned segment — and at seg=50 the price halved, making blackmail profitable.
The others' near-1.0 acceptance is a rational best response: rejecting A5's 3.0
only delays the inevitable and returns the pen to A5.

- **The dictatorship is oddly benign on aggregates**: welfare 2377±54 (lowest
  variance of any run), equality 0.925, ratio 1.046, river 139, only 42/1000
  steps uncontracted — a *stable extractive institution* whose rent is largely
  dissipated by cleaner entry anyway. Compare seg=100 holdout (θ̄≈1.55,
  interior): the exploit is specifically the CHEAP-DELAY × holdout interaction,
  not holdout itself.
- **Belief curves elsewhere finally have the right shapes**: harvesters' gaps
  decrease in θ, cleaners' increase with reservation crossings at θ≈2.45–2.52 —
  the sharpest value heads yet (shorter segments = more value data). A5's own
  heads are meaningless: as responder it never experiences a lock (its
  rejections prevent them), so its lock head trains on nothing — the
  entrenchment is self-sealing in data as well as strategy.
- **Protocol-design lesson for the write-up**: recognition-by-holdout requires
  delay to be expensive; the segment length is not a nuisance parameter but the
  price of proposal power, and below some threshold the mechanism flips from
  "rejecters counter" to "rejecters rule". A mixed recognition rule (pen to a
  rejecter with prob q < 1, else random) is the natural tax on the monopoly —
  held in reserve.

## D. The vote-blindness defect and the fix (2026-08-12)

- **Root cause of both degenerate equilibria:** features are built *before* the
  proposer acts, so votes are conditioned on (round, proposer identity, last
  round's dead offer, own stats) but **not on the offer currently on the table**.
  A threshold strategy — "reject θ below my reservation value" — is
  unrepresentable; the only expressible vote policies are proposer-conditioned
  (seed 42) or constant (seed 55). Under a capped ceiling, blanket rejection of
  others' unseeable offers is a coherent best constant, which is why the veto
  dictator emerged exactly when the ceiling was lowered to 1.0. **[OURS]**
- **Fix commissioned:** two-pass rounds (proposer emits θ, responders and their
  critic then see it), last-round vote-pattern features, advantage normalization
  over active rounds only, rollout-time exploration floor on the vote. Post-fix
  success signatures: accept rate below 1 concentrated on low offers; round
  histogram spreading past R0; offered θ rising with round; cleaner:harvester
  ratio up from 0.890.
- **Write-up caution:** the docstring argument that a feedforward net suffices
  because "the SPE is Markov in (round, proposer)" is Opus-written and assumes
  complete information. With private payoff histories and co-learning opponents,
  history carries information the Markov state discards — the seed-42 cleaner's
  history-dependent gameplay coupling is a live counterexample. Treat the claim
  as a design heuristic, not a theorem about this system.

## E. Planned evaluation and levers (for later sections)

- **ANAC/Genius scripted probes** (Boulware / Conceder / linear time-dependent
  concession proposers; threshold responders) as *evaluation* opponents and case
  studies: how does the learned bargainer fare against a fixed tough negotiator?
  Exploitability-style evidence, stronger than self-play numbers alone. Intended
  for a separate case-study section of the dissertation.
- **Forced-null episodes in joint training** (Leon's proposal): enforce the null
  contract for a fraction of episodes/segments so gameplay learns distinct
  disagreement behaviour. Motivation: the null arm currently still cleans at
  0.94/step because gameplay rarely saw θ=0 — the asymmetric-decay disagreement
  path the bargaining argument relies on is not yet realised in behaviour. This
  is the joint-mode analogue of the phase-1 NULL_CONTRACT_FRAC conditioning.
- **FTRL / computed-equilibrium arm over the frozen subgame**: with gameplay
  frozen and θ discretized, the bargaining layer becomes a small repeated game —
  payoffs precomputable by rollout, tabular no-regret dynamics (FTRL/Hedge)
  runnable at scale, and the SPE computable outright by backward induction.
  Learned (PPO) vs no-regret (FTRL) vs computed (induction / Nash / KS via
  solver.py) on the same game would be a strong methods contribution.
- **Reserve levers if round-0 unanimity survives the fix:** recognition-by-holdout
  (rejecters draw the next proposal slot), curriculum opponents (scripted tough
  responders / lowball proposers, annealed out), sequential polled voting (kills
  the all-reject equilibrium and non-pivotality), a grounded counteroffer signal
  attached to rejections.
- **COMA-style counterfactual vote credit — no longer in reserve, built
  2026-08-12** (`BARGAIN_VOTE_ADVANTAGE=counterfactual`). The [0, 3] run above
  isolated credit assignment as the remaining binding constraint, so this was
  commissioned rather than held: each vote is now credited with
  pivotality × (value if this offer locks − value of continuing), from two learned
  branch heads, and non-pivotal votes are masked out of the policy gradient
  entirely. Cheap under unanimity exactly as anticipated — one comparison, no
  marginalisation. Shipped alongside the `BARGAIN_ACCEPT_BIAS` 1.0 → 0.5 ablation,
  since probes and the null-pass rule now do most of what the accept prior was
  for, while the prior itself leans toward the failure under study. Success
  signature to check on the next run: θ-sweep curves with genuine 0.5 crossings
  (cleaners low, harvesters high at [0, 3]), and `joint/cf/gap` turning negative on
  lowballs. Failure signature worth naming in advance: p(accept) flat while the
  believed gap is correctly signed would mean the branch heads learned the right
  thing and the policy still did not act on it.

## E2. Reporting mechanism, first run (runs/reporting_tests/v1) — 2026-08-13

**[OURS, 40-episode eval; gae/rotate/seg100/[0,3] substrate; p=0.25, λ=2, cap 20
— deliberately on the lying side of λ\*=3. No sidecar came back; params passed by
flag.]**

- **Fraud emerged from honest initialisation, in every agent.** Mean overclaim
  15.1 of cap 20; harvesters claim ~10× their true cleaning (true ~1.3–1.8
  cells/window, claimed +14–16); cleaners (true ~25) inflate by the same ~15.
  Caught rate ≈ audit p (0.22–0.30 vs 0.25) as it must be. With per-unit lying
  EV = θ(1−p(1+λ)) = +0.25θ, the optimum is the cap; policies sit at ~75% of it
  (Gaussian spread + possibly still climbing). The Becker calculus is being
  found by learning — the instrument works.
- **The costs**: welfare 1747±509 vs the paired no-reporting control's 2555
  (−32%), equality 0.626 vs 0.910. Three channels: (a) audit-lottery variance —
  identical strategies, wildly different returns (A6 153 vs A3 332); (b) low-θ
  locks became catastrophic (θ≈0.1–0.2 episodes at welfare 226–366 — with claim
  income ∝ θ, near-null contracts kill both channels at once); (c) even θ=3
  episodes lost ~400 vs control — suspected TRAINING ARTIFACT: settlements land
  as ±60-scale lumps on window-boundary steps inside the gameplay reward stream
  (vs ~1/step apples), injecting credit noise into harvest policies. (c) is an
  implementation cost of the rung-1 reward routing, not a finding about fraud —
  keep the two separate in the write-up.
- Negotiation itself tolerated the enforcement layer: agreement 40/40, θ̄=2.37,
  R0 acceptance 0.80 — the contract and reporting layers compose.
- **Fraud scales with θ** (offline sweep of the claim heads): mean overclaim
  rises monotonically ~9.6 → ~16 as θ goes 0.25 → 3, for every agent. The
  *rational optimum is the cap at any θ>0*, so the slope is a learning-dynamics
  fingerprint, not an equilibrium property: the REINFORCE push per unit of
  overclaim is θ(1−p(1+λ)) = 0.25θ, so policies climbed fastest where lying
  pays most — the same slopes-before-levels signature the vote head showed.
  Prediction: longer training flattens the curve toward the cap everywhere.
- **Cleaners overclaim slightly more than harvesters at equal θ** (+0.7–1.0).
  Note "having cover" confers zero advantage under exogenous audits (detection
  is mechanical, not plausibility-based) — this measurement is the baseline
  against which rung-2 strategic auditing, where cover SHOULD matter, will be
  read.
- **Seven independent learners converged to near-identical fraud curves**
  (spread <0.2 at every θ) — the bandit is symmetric across agents, so this is
  an internal replication of the Becker response, seven times in one run.
  Sanity: the sweep at the run's θ̄=2.37 gives ~15, matching the eval's 15.05.
- **Next**: the honesty-side twin (λ=6) to complete the Becker pair; then a
  (p, λ) grid. If welfare stays depressed at λ=6 (where lying ≈ 0 and
  settlements ≈ 0), the noise explanation for (c) weakens and something real is
  going on. Figure idea: pool all reporting runs and plot learned overclaim
  against the per-unit incentive θ(1−p(1+λ)) — the "price of fraud" curve.

## F. Standing methodological cautions

- **Contract range must travel with the weights.** A mismatched range rescales θ
  through both the observation and the proposal unsquash; every number comes out
  wrong while looking normal. Already invalidated one full three-arm comparison
  (figures recorded in [findings.md](findings.md) so they're recognisable if they
  resurface). Provenance sidecars (`.run.yaml`) now prevent this.
- **`apple_reward` defaults to `num_agents`**: any eval harness must set 1.0
  explicitly or returns are 7× off.
- **Individual reward is the point**: under shared rewards the dilemma is removed
  by construction; nothing under `reward=common` says anything about cooperation.

## Positioning against adjacent work (for related-work section)

Learned-redistribution mechanisms in social dilemmas now include planner-imposed
taxes (LOPT, NeurIPS 2025), unilateral peer incentives (LIO, arXiv:2006.06051 —
also uses Cleanup), precomputed minimal transfer contracts (arXiv:2310.12928), and
principal-agent RL (arXiv:2407.18074). The differentiator here: the *terms* of
redistribution are negotiated by the affected parties themselves, under a protocol
whose disagreement path is priced by the commons — neither planner-imposed nor
unilateral, and requiring no fairness axiom.
