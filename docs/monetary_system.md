# Monetary system in Clean Up (`pay_mode`)

A minimal agent-controlled payment mechanism for studying whether the ability to
wire reward to other agents lets cooperation emerge between specialists (river
cleaners vs. apple harvesters).

## Mechanism

- New 10th action `Actions.pay` in `socialjax/environments/cleanup/clean_up.py`.
- An agent taking `pay` sends `pay_amount` (default **1.0**) of this step's reward to
  the **nearest other agent within Chebyshev distance `pay_radius`** (default
  `obs_size // 2 = 5`, approximating the agent's field of view). Ties break to the
  lowest agent index. No agent in range → the action does nothing.
- Transfers are **zero-sum by construction** (sender −1, receiver +1, same step) and
  **stateless** — there is no persistent balance, so no observation change is needed.
- Senders *can* go net-negative on a step (pay while earning nothing): paying is a
  genuine sacrifice, which is the point.
- Implementation: `compute_pay_transfers()` (module-level, unit-tested in isolation),
  called from `_step`. `pay_mode` is static at trace time, so `"off"` compiles to the
  exact original reward graph.

## The three experimental conditions

| Condition | `ENV_KWARGS` | Action space | What it isolates |
|---|---|---|---|
| baseline | *(none — default `pay_mode=off`)* | 9 | original Clean Up, comparable to all previous results |
| placebo | `+ENV_KWARGS.pay_mode=noop` | 10 | action-space-size effects only: `pay` selectable but transfers nothing |
| treatment | `+ENV_KWARGS.pay_mode=on` | 10 | the money itself |

If baseline ≈ placebo and treatment differs from both, the difference is attributable
to the payment mechanism, not to the extra action.

Run all conditions with `reward=individual` (under `reward=common` a private transfer
is meaningless — everyone already shares one summed reward) and
`PARAMETER_SHARING=False` (independent networks per agent, so role specialization
isn't suppressed by weight sharing).

## Metrics (logged to WandB automatically via the env `info` dict)

- `pay_attempts` — fraction of agent-steps choosing `pay`
- `pay_executed` — fraction of agent-steps where `pay` had a valid in-range target
- `pay_volume` — total reward transferred per step (always 0 under `noop`)
- plus the existing `original_rewards`, `clean_action_info`, `cleaned_water`, …

## Tests

```bash
python tests/test_pay_mechanism.py        # 15 checks: zero-sum, targeting, jit, env integration
```

Run these after any change to the mechanism and before launching cloud runs. They
exercise the full jitted `_step` — if they pass on CPU, the same XLA graph lowers on GPU.

## Local smoke test (CPU, ~2 min, run before every GCP launch)

```bash
python algorithms/train.py --algo IPPO --env cleanup reward=individual \
  WANDB_MODE=disabled PARAMETER_SHARING=False \
  NUM_ENVS=4 NUM_STEPS=32 NUM_MINIBATCHES=2 UPDATE_EPOCHS=1 TOTAL_TIMESTEPS=256 \
  GIF_NUM_FRAMES=8 ENV_KWARGS.num_agents=3 ENV_KWARGS.num_inner_steps=64 \
  +ENV_KWARGS.pay_mode=on
```

## Full runs on GCP (per condition; sweep seeds with -m)

```bash
# treatment
python algorithms/train.py --algo IPPO --env cleanup reward=individual \
  PARAMETER_SHARING=False +ENV_KWARGS.pay_mode=on -m SEED=42,52,62

# placebo
python algorithms/train.py --algo IPPO --env cleanup reward=individual \
  PARAMETER_SHARING=False +ENV_KWARGS.pay_mode=noop -m SEED=42,52,62

# baseline
python algorithms/train.py --algo IPPO --env cleanup reward=individual \
  PARAMETER_SHARING=False -m SEED=42,52,62
```

Defaults from `ippo_base.yaml` apply (1e8 timesteps, 256 envs, WandB on). Optional
knobs: `+ENV_KWARGS.pay_amount=2.0`, `+ENV_KWARGS.pay_radius=3`.

Note: with `PARAMETER_SHARING=False`, checkpoints are saved **once, at the end of
training** (`checkpoints/individual/<env>_seed<S>_reward_individual_<agent>.pkl`).
Prefer non-preemptible instances for long runs, or add periodic checkpointing first.

## Visualizing a trained run locally

```bash
python viz/interactive_viewer.py --env clean_up --num_agents 7 \
  --env-kwarg pay_mode=on \
  --checkpoint 'checkpoints/individual/clean_up_seed42_reward_individual_*.pkl'
```

The glob (quoted!) loads one `.pkl` per agent, sorted → agent 0..N−1. The env must be
created with the same `pay_mode`/`num_agents` the checkpoint was trained with, or the
network's action head won't match.
