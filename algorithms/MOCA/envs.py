"""Which environment a contracting run is played on, and what changes with it.

The contracting machinery -- the scalar contract space, phase 1's P(Theta), the
proposal/vote protocols, the solver -- is environment-independent by construction.
Three things are not, and they are collected here rather than scattered through the
training loop:

  * the CONTRACT SPACE, i.e. what the bargained scalar buys (contracts.py holds the
    mechanisms; this says which one belongs to which env);
  * the BEHAVIOUR METRICS, the per-env series that answer "did the contract change
    what agents actually did", without which welfare moving is uninterpretable;
  * the DEFAULT theta RANGE, which is a property of the environment's economy and
    differs by a factor of 50 between Clean Up and Harvest.

Adding an environment means adding an entry here and a contract class in
contracts.py -- not touching the training loop.

Registered here does NOT mean supported by every arm. Joint Rubinstein bargaining
(TRAINING_MODE=joint) reads the river state and a cleaning window that only Clean Up
has; `check_arm` is what refuses that combination up front instead of failing on a
missing info key several thousand compiled lines later.
"""
from typing import Dict, NamedTuple, Tuple

from algorithms.MOCA import contracts


class EnvSpec(NamedTuple):
    """Everything the contracting loop needs to know about one environment.

    Attributes:
        env_name: the socialjax registry id.
        contract_space: key into contracts.CONTRACT_SPACES.
        num_agents: the agent count the published results use, and the default here.
        reward_scale_kwarg: the env kwarg that sets one unit of primary reward. Every
            contract range is quoted against a UNIT reward, and all three envs default
            this to num_agents (so that individual- and shared-reward arms carry equal
            total reward mass), which would silently rescale theta by a factor of N.
            Named so `check_reward_scale` can say which knob to set.
        behaviour_metrics: info fields, without the _mean/_std suffix, that say
            whether behaviour moved. First entry is the CONTRACTED ACT -- the quantity
            the contract prices -- and is what the phase-1 null/contracted split is
            reported on.
        commons_metric: the stock or state of the common resource. Welfare can only
            move if behaviour does, and behaviour is only worth anything if it feeds
            back into the commons, so this is the series that connects the two.
        progress_metric: info field the console progress line reports.
        act_label: short name for the contracted act in the phase-1 null-vs-contracted
            series (<label>_null / <label>_contracted / <label>_gap). Separate from
            the info key so Clean Up keeps the series names its existing runs and
            wandb views already use.
    """
    env_name: str
    contract_space: str
    num_agents: int
    reward_scale_kwarg: str
    behaviour_metrics: Tuple[str, ...]
    commons_metric: str
    progress_metric: str
    act_label: str

    @property
    def contracted_act(self) -> str:
        """The info field the contract conditions transfers on."""
        return self.behaviour_metrics[0]

    @property
    def contract_range(self) -> Tuple[float, float]:
        """The paper's theta range for this environment."""
        return contracts.CONTRACT_SPACES[self.contract_space].DEFAULT_RANGE


ENV_SPECS: Dict[str, EnvSpec] = {
    "clean_up": EnvSpec(
        env_name="clean_up",
        contract_space="cleanup",
        num_agents=7,
        reward_scale_kwarg="apple_reward",
        # Cleaning is the public good the contract subsidises; the std across agents
        # is the division of labour that produces, which pooled means hide entirely.
        behaviour_metrics=("cleaned_by_agent",),
        commons_metric="waste_cleared",
        progress_metric="shaped_rewards",
        act_label="cleaned",
    ),
    "harvest_common_open": EnvSpec(
        env_name="harvest_common_open",
        contract_space="harvest",
        num_agents=7,
        reward_scale_kwarg="apple_reward",
        # low_density_eaten is what the contract charges for; eaten_apples is the
        # denominator. Both are needed: a contract can cut the first either by
        # stopping depletion (the point) or by stopping harvesting altogether (a
        # welfare loss dressed up as success), and only their ratio tells them apart.
        behaviour_metrics=("low_density_eaten", "eaten_apples"),
        commons_metric="apple_stock",
        progress_metric="shaped_rewards",
        # "thin_patch_eats", not "depleting_eats": the series counts a specific,
        # checkable event -- ate an apple that had fewer than `low_density_threshold`
        # apples in the 21 cells around it -- and "depleting" named a consequence
        # instead, which invited reading it as "apples removed from the commons"
        # (that is `eaten_apples`).
        act_label="thin_patch_eats",
    ),
    "coin_game": EnvSpec(
        env_name="coin_game",
        contract_space="coin_game",
        num_agents=2,
        reward_scale_kwarg="coin_reward",
        # Same denominator argument: a contract that stops all coin collection has
        # driven theft to zero without helping anybody.
        behaviour_metrics=("stolen_by_agent", "coins_taken"),
        # No stock to deplete here -- coins respawn on collection -- so the commons
        # series is own-colour collection, the cooperative act theft crowds out.
        commons_metric="eat_own_coins",
        # coin_game's `shaped_rewards` is the SUMMED reward broadcast to every agent
        # in the individual-reward branch, so it would report N x welfare rather than
        # a per-agent return. `original_rewards` there is the per-agent reward the env
        # actually returns. Left as found rather than corrected, because IPPO runs on
        # this env have been logging it for months and a silent change of meaning
        # mid-project is worse than an asymmetry recorded in one place.
        progress_metric="original_rewards",
        act_label="stolen",
    ),
    "coin_game_n": EnvSpec(
        env_name="coin_game_n",
        contract_space="coin_game",
        # Seven, to sit alongside Clean Up and Harvest. The two-player `coin_game`
        # stays registered and is NOT superseded: with one responder its bargaining
        # arm is literal Rubinstein alternating offers, the only place in this project
        # where the protocol matches the theory exactly rather than approximating it.
        num_agents=7,
        reward_scale_kwarg="coin_reward",
        # eat_own_coins is a THIRD entry rather than the commons metric it is in the
        # two-player game, because it is needed here for a different reason: with the
        # -2 penalty, welfare reduces exactly to
        #     welfare = own coins collected - coins stolen
        # (every theft pays its taker +1 and costs its victim -2, and each theft has
        # exactly one victim). Without own-coin collection logged, a welfare number
        # cannot be decomposed into "more cooperation" versus "less theft", which are
        # the two ways it can move and are not the same result. Entries after the
        # first two are logged and nothing else -- the contracted act and the
        # denominator the _share ratio uses are still [0] and [1].
        behaviour_metrics=("stolen_by_agent", "coins_taken", "eat_own_coins"),
        # A real grid-wide stock, uniform across agents -- unlike `coin_game`, whose
        # eat_own_coins is genuinely per-agent and is therefore read as agent 0's
        # value alone by the bargaining rollout.
        commons_metric="coins_on_grid",
        # Per-agent either way here (coin_game_n's shaped_rewards is not the summed
        # broadcast the two-player env logs), but kept as original_rewards so the two
        # coin environments report the same series.
        progress_metric="original_rewards",
        act_label="stolen",
    ),
}

#: Bargaining protocols available per environment. `alternating` -- one proposer, a
#: vote, a quorum -- is the protocol the renegotiation results are stated in, and it
#: runs everywhere. `median` (every agent names a theta simultaneously, the middle
#: ask binds) is a Clean Up experiment: its whole argument rests on ~4 harvesters
#: among 7 agents putting the median near the welfare optimum, which is a property of
#: that environment's role split and not of contracting.
#: Contract spaces available per environment. The first is the default and the one
#: `EnvSpec.contract_space` names; the rest are alternatives selectable with
#: CONTRACT_SPACE. An environment has one space in the reference -- `harvest_density`
#: is this project's, bargaining the density threshold that the published Harvest
#: contract fixes at construction, and it is listed here rather than replacing
#: `harvest` so the one-dimensional runs stay loadable and the comparison is a flag.
CONTRACT_SPACES_BY_ENV = {
    "clean_up": ("cleanup",),
    "harvest_common_open": ("harvest", "harvest_density"),
    "coin_game": ("coin_game",),
    "coin_game_n": ("coin_game",),
}

BARGAIN_PROTOCOLS_BY_ENV = {
    "clean_up": ("alternating", "median"),
    "harvest_common_open": ("alternating",),
    "coin_game": ("alternating",),
    "coin_game_n": ("alternating",),
}


def commons_scale(spec: EnvSpec, env) -> float:
    """Divisor that puts the commons state at roughly unit range in the bargaining
    state. Read off the environment rather than hardcoded, so a run on a different
    map does not silently feed the policy an out-of-range feature.

    Each is the maximum the commons metric can take: every cell clear on Clean Up,
    every spawn point holding an apple on Harvest, and on the Coin Game the metric
    is a per-step reward flow rather than a stock, so its natural unit is one coin.
    """
    if spec.env_name == "clean_up":
        return float(env.GRID_SIZE_ROW * env.GRID_SIZE_COL)
    if spec.env_name == "harvest_common_open":
        return float(len(env.SPAWNS_APPLE))
    if spec.env_name == "coin_game_n":
        # Coins on the grid IS a stock here, so it scales like the other two: every
        # spawn cell holding a coin at once.
        return float(len(env.SPAWNS_COIN))
    return 1.0


def spec_for(env_name: str) -> EnvSpec:
    """The EnvSpec for a socialjax env id, or a message naming what is available."""
    try:
        return ENV_SPECS[env_name]
    except KeyError:
        raise ValueError(
            f"no contracting spec for environment {env_name!r}. Contracting is "
            f"implemented for: {', '.join(sorted(ENV_SPECS))}. Adding one means an "
            f"entry in algorithms/MOCA/envs.py and a contract class in "
            f"algorithms/MOCA/contracts.py."
        ) from None


def check_arm(spec: EnvSpec, phase2_mode: str, training_mode: str,
              config: dict = None) -> None:
    """Refuse env/arm combinations that are not implemented, at config time.

    The failure this exists to prevent is not a crash -- it is a code path reading
    `cleaned_by_agent` on an environment that has no cleaning, which surfaces as a
    KeyError thousands of traced lines from the config that caused it.

    The renegotiating bargaining arm (TRAINING_MODE=joint, PHASE2_MODE=bargain) runs
    on every environment. What stays Clean Up-only is the machinery underneath it
    that is genuinely about cleaning: the harvest-tax contract kind, the
    claims-and-audits channel (an overclaim is an overclaim OF CLEANING), and the
    median protocol.
    """
    config = config or {}
    if training_mode == "joint" and phase2_mode == "bargain":
        protocol = config.get("BARGAIN_PROTOCOL", "alternating")
        allowed = BARGAIN_PROTOCOLS_BY_ENV[spec.env_name]
        if protocol not in allowed:
            raise ValueError(
                f"BARGAIN_PROTOCOL={protocol!r} is not available on "
                f"{spec.env_name!r} (available: {', '.join(allowed)}). The median "
                f"mechanism's argument is that with ~4 harvesters among 7 agents the "
                f"median ask tracks the harvester ideal, which the welfare(theta) "
                f"curve puts near its optimum -- a property of Clean Up's role split, "
                f"not of contracting. Use BARGAIN_PROTOCOL=alternating.")
        if config.get("REPORT_ENABLE") and spec.env_name != "clean_up":
            raise ValueError(
                f"REPORT_ENABLE is Clean Up only: a claim is an overclaim of CLEANING, "
                f"filed against a per-cell wage. {spec.env_name!r} has no such wage "
                f"to inflate. See docs/reporting.md.")
    if config.get("CONTRACT_KIND", "clean_wage") != "clean_wage" \
            and spec.env_name != "clean_up":
        raise ValueError(
            f"CONTRACT_KIND is a Clean Up setting; {spec.env_name!r} has one contract "
            f"space. Got {config.get('CONTRACT_KIND')!r}.")


def check_reward_scale(spec: EnvSpec, env_kwargs: dict) -> None:
    """Warn if theta is being quoted against a non-unit reward.

    Every contract range in the literature is per unit of primary reward: 0.2 per
    waste cell against an apple worth 1, 10 per depleting harvest against an apple
    worth 1. All three environments here default their reward scale to num_agents
    instead, so leaving it unset multiplies every payoff by N while theta stays put --
    which does not error, does not look wrong in any logged series, and quietly makes
    the whole contract range N times too weak.
    """
    scale = env_kwargs.get(spec.reward_scale_kwarg)
    if scale is None:
        print(
            f"[MOCA warning] ENV_KWARGS.{spec.reward_scale_kwarg} is unset, so "
            f"{spec.env_name} defaults it to num_agents={spec.num_agents}. Contract "
            f"ranges are quoted against a UNIT reward, so theta is effectively "
            f"{spec.num_agents}x weaker than the range says. Set "
            f"{spec.reward_scale_kwarg}: 1.0.",
            flush=True,
        )
    elif float(scale) != 1.0:
        print(
            f"[MOCA warning] ENV_KWARGS.{spec.reward_scale_kwarg}="
            f"{scale} != 1.0: the contract range "
            f"{spec.contract_range} is stated against a unit reward and does not "
            f"carry over to this scale.",
            flush=True,
        )
