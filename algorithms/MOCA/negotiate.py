"""The learned contracting stage -- MOCA Algorithm 1's phase 2.

This is `SeparateContractNegotiateStage` from the reference implementation
(github.com/Algorithmic-Alignment-Lab/contracts,
environments/two_stage_train.py), the contracting game the paper's Algorithm 1
describes. Note it is NOT what produced the paper's Cleanup numbers -- that config
sets `"solver": true`, which disables this stage (see algorithms/MOCA/solver.py).
It is the algorithm the authors ran on their other domains.

The negotiation is a TWO-STEP episode (`config_negotiate['horizon'] = 2`) played
before each game episode, with the gameplay policy frozen:

  step 0, contract_state = 2 (PROPOSE)
      Every agent emits an action, but only agent 0's contract components are
      read: `self.params = {key: acts['a0'][:-1] for key in acts.keys()}`.
      The proposer is FIXED at agent 0, not sampled -- the paper's "one crucial
      assumption in our analysis is that a single agent proposes contracts".
      Reward is 0 for everyone.

  step 1, contract_state = 3 (AGREE)
      nu agents are drawn from the NON-proposers and the product of their accept
      probabilities is the probability the contract takes force::

          if self.num_agents > 3:
              chosen_agents = random.sample(range(1, self.num_agents), 2)
          else:
              chosen_agents = range(1, self.num_agents)
          for term in [acts['a'+str(i)][-1] for i in chosen_agents]:
              prod_prob *= term
          r = random.random()
          decision = 1 if r < prod_prob else 0

      On rejection the contract becomes null. The frozen policy then plays the
      whole game episode under whichever contract resulted, and the agents'
      episode returns are the reward for this step.

Because reward arrives only at step 1, the proposer's action is credited through
the value function across one discount step -- this is PPO on a two-step MDP, not
a bandit.
"""
import jax
import jax.numpy as jnp


def unsquash(raw, low, high):
    """Map a normalised action onto [low, high], as RLlib's normalize_actions does.

    RLlib's `unsquash_action` computes `low + (a + 1) * (high - low) / 2` and then
    clips to the bounds. The clip is applied to the ENVIRONMENT-facing action only;
    PPO keeps the log-probability of the raw Gaussian sample, so the gradient still
    flows for samples that landed outside the box.
    """
    scaled = low + (raw + 1.0) * (high - low) / 2.0
    return jnp.clip(scaled, low, high)


def default_nu(num_agents: int) -> int:
    """Number of non-proposers polled, following the reference's branch exactly.

    `random.sample(range(1, num_agents), 2)` when num_agents > 3, otherwise every
    non-proposer. With <= 3 agents there are at most 2 non-proposers anyway, so the
    branch only bites for larger games -- where it is what keeps the acceptance
    probability from collapsing as a product of many near-even terms.
    """
    return 2 if num_agents > 3 else max(num_agents - 1, 1)


def sample_voters(key, num_agents: int, nu: int, num_envs: int):
    """Boolean (num_agents, num_envs) mask of who is polled, per env.

    Sampling is without replacement from agents 1..N-1; agent 0 is the fixed
    proposer and is never a voter.
    """
    u = jax.random.uniform(key, (num_agents, num_envs))
    u = u.at[0].set(2.0)                                  # proposer never sampled
    rank = jnp.argsort(jnp.argsort(u, axis=0), axis=0)
    return rank < nu


def acceptance(key, accept_probs, voter_mask):
    """Whether the contract takes force, per env.

    Args:
        accept_probs: (num_agents, num_envs) each agent's accept probability, the
            last component of its action.
        voter_mask: (num_agents, num_envs) which agents were polled.

    Returns:
        accepted: (num_envs,) bool.
        prod_prob: (num_envs,) the product of the polled agents' probabilities.
    """
    # Unpolled agents contribute a factor of 1, leaving the product over voters.
    prod_prob = jnp.prod(jnp.where(voter_mask, accept_probs, 1.0), axis=0)
    return jax.random.uniform(key, prod_prob.shape) < prod_prob, prod_prob


def two_step_gae(rewards, values, gamma, gae_lambda):
    """GAE over the two-step negotiation episode.

    Args:
        rewards: (2, ...) step-0 reward is 0; step-1 reward is the episode return.
        values: (2, ...) critic at the proposal and agreement states.

    Returns:
        advantages, targets, each (2, ...).
    """
    # The episode terminates after step 1, so the bootstrap value is 0.
    delta_1 = rewards[1] - values[1]
    adv_1 = delta_1
    delta_0 = rewards[0] + gamma * values[1] - values[0]
    adv_0 = delta_0 + gamma * gae_lambda * adv_1
    advantages = jnp.stack([adv_0, adv_1])
    return advantages, advantages + values
