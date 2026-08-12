"""Networks for MOCA: a contract-conditioned gameplay policy plus the two
contracting-stage policies (proposal and voting).

The reference implementation exposes the contract to a convolutional agent as a
separate observation key -- gym.spaces.Dict({'image': ..., 'contract': ...}) --
whose 'contract' entry is np.concatenate((params, [stage])). RLlib's default model
then embeds the image with a CNN and concatenates the remaining (contract) features
before the policy head. ContractActorCritic reproduces exactly that structure on
top of this repo's existing CNN trunk, so the gameplay policy conditions on theta
without changing the image observation itself.
"""
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

from algorithms.utils.networks import CNN


class ContractActorCritic(nn.Module):
    """Actor-critic conditioned on the active contract.

    Mirrors algorithms/utils/networks.py::ActorCritic (same CNN trunk, same head
    sizes and initialisers) but concatenates the contract feature vector onto the
    CNN embedding before the actor and critic heads.

    Conditioning on theta is not optional in MOCA: Phase 1 trains this policy across
    randomly drawn contracts precisely so that it represents the whole family
    {pi(.|s, theta)} and hence an unbiased V_i(s_0, theta) for every theta. A policy
    blind to theta could not represent that family at all.

    Attributes:
        action_dim: number of discrete actions.
        activation: "relu" or "tanh".
    """
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, contract):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        embedding = CNN(self.activation)(x)
        # Contract features enter alongside the visual embedding, matching the
        # reference Dict-observation layout ('image' + 'contract').
        embedding = jnp.concatenate([embedding, contract], axis=-1)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class NegotiationActorCritic(nn.Module):
    """The contracting-stage policy of the reference implementation's stage 2.

    `SeparateContractNegotiateStage` gives the negotiation stage a single CONTINUOUS
    action space per agent::

        self.action_space = gym.spaces.Box(
            low=np.concatenate((self.contract_low, np.array([0.0]))),
            high=np.concatenate((self.contract_high, np.array([1.0]))))

    i.e. one vector holding the contract parameters followed by an accept
    probability. Which part of it is read depends on the stage: at the proposal
    step the env takes `acts['a0'][:-1]` as the contract, and at the agreement step
    it takes `acts['a'+str(i)][-1]` from the sampled voters as accept probabilities.
    Every agent emits the full vector at both steps; the unused components simply do
    not affect the environment, exactly as in RLlib.

    A fresh network, not the gameplay policy: the reference trains a separate PPO
    trainer on the negotiation env and only loads the frozen subgame policy inside
    it to roll out episodes.

    Actions are emitted in normalised space and mapped onto the action bounds by
    `negotiate.unsquash`, which is RLlib's `normalize_actions=True` behaviour --
    the reason a unit-scale Gaussian is a sensible initialisation even though the
    contract range here is only [0, 0.2].

    Attributes:
        action_dim: contract parameter dims + 1 (the accept probability).
        activation: "relu" or "tanh".
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, contract):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        embedding = CNN(self.activation)(x)
        embedding = jnp.concatenate([embedding, contract], axis=-1)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        # State-independent log-std, as in RLlib's DiagGaussian: zero-initialised,
        # so the initial policy is unit-variance in normalised action space and
        # therefore spreads proposals across the whole contract range.
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ProposalPolicy(nn.Module):
    """pi_i(i, 0): the proposing agent's distribution over contracts.

    The contracting state (i, 0) carries no game state -- proposals are made before
    play begins, from a fixed initial-state distribution -- so the policy has no
    input and reduces to a learnable set of logits over the discretised contract
    grid. That makes Phase 2 an episode-level bandit over contracts, which is
    exactly what MOCA's frozen-subgame construction turns the problem into.

    Attributes:
        num_contracts: size of the contract grid (K).
    """
    num_contracts: int

    @nn.compact
    def __call__(self):
        logits = self.param(
            "proposal_logits", nn.initializers.zeros, (self.num_contracts,)
        )
        return distrax.Categorical(logits=logits)


class VotingPolicy(nn.Module):
    """pi_j(i, theta): agent j's accept/reject decision on a proposed contract.

    Conditions on who proposed (one-hot) and the normalised contract value, so an
    agent can learn to accept contracts that benefit it and veto ones that do not.
    Unanimous acceptance is required for the contract to take force, which is what
    makes an accepted contract individually rational for every signatory.

    Attributes:
        num_agents: N, for the proposer one-hot.
        hidden: hidden layer width.
    """
    num_agents: int
    hidden: int = 32

    @nn.compact
    def __call__(self, proposer_onehot, theta_norm):
        x = jnp.concatenate([proposer_onehot, theta_norm[..., None]], axis=-1)
        x = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        logits = nn.Dense(2, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(x)  # [reject, accept]
        return distrax.Categorical(logits=logits)


class BargainingActorCritic(nn.Module):
    """The Rubinstein bargaining policy: an MLP over a compact bargaining state.

    Deliberately NOT the CNN trunk the reference uses for its negotiation stage. Two
    reasons, one structural and one empirical.

    Structural: the agent observation is an egocentric OBS_SIZE x OBS_SIZE window
    (11x11 on a 19x28 grid), which cannot represent the round index, whose turn it
    is, the standing offer, or the state of the river as a whole. The reference gets
    away with feeding it to a negotiation head because its game is one shot at s_0,
    where every episode looks alike; a multi-round game whose decisions condition on
    history cannot. Some state augmentation is therefore forced, not optional.

    Empirical: this is what the deep-RL bargaining literature does. Agents are given
    "its utility function, the offer given by the opponent, its previous offer, and
    agent ID", encoded through "an OfferMLP (multi-layer perceptron with ReLU
    activation), an agent embedding ... and a turn embedding", with "the scalar
    representing the current round of bargaining ... included in the state, as is
    which agent was selected to propose and whether the current agent is proposing or
    responding" (RLBOA and the multi-issue negotiation line of work). Cao et al.
    (2018) likewise embed structured game state rather than raw perception.

    And feedforward suffices: the SPE of a FINITE alternating-offers game is Markov
    in (round, proposer), both of which are inputs here, so no recurrence is needed
    to represent the equilibrium strategy.

    Three heads, because the reference's single Gaussian over [theta, accept_prob]
    conflates two different decisions and mis-specifies one of them: its accept
    "probability" is a clipped Gaussian coordinate whose log-probability is the
    Gaussian's, not a Bernoulli's, so the gradient does not correspond to the
    accept/reject choice being made.

    Attributes:
        hidden: width of the two shared layers.
        activation: "relu" or "tanh".
        accept_bias: initial logit added to ACCEPT. Unanimity among 6 responders at
            an unbiased initialisation fires with probability 0.5**6 = 1.6%, so
            without a positive prior the contract is almost never in force and the
            proposal head sees no signal to learn from. Annealing the quorum is the
            other lever; this one is free.
        aux_heads: build the two BRANCH value heads the counterfactual vote
            advantage needs (`lock_value`, `cont_value`; see bargain.py). Off by
            default and skipped entirely when off, so the parameter tree, the
            initialisation RNG and therefore every existing checkpoint are
            bit-for-bit what they were before the heads existed. They are appended
            AFTER the critic so the automatic Dense_N numbering of everything above
            is untouched, and named, so a loader can detect them by key.
    """
    hidden: int = 64
    activation: str = "relu"
    accept_bias: float = 1.0
    aux_heads: bool = False

    @nn.compact
    def __call__(self, x, return_aux: bool = False):
        act = nn.relu if self.activation == "relu" else nn.tanh
        h = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        h = act(h)
        h = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        h = act(h)

        # Proposal: a scalar contract in normalised space, unsquashed onto
        # [low, high] by negotiate.unsquash exactly as the reference does. Kept
        # Gaussian with a state-independent log-std, matching RLlib's DiagGaussian.
        theta_mean = nn.Dense(1, kernel_init=orthogonal(0.01),
                              bias_init=constant(0.0))(h)
        log_std = self.param("log_std", nn.initializers.zeros, (1,))
        pi_theta = distrax.MultivariateNormalDiag(theta_mean, jnp.exp(log_std))

        # Vote: a real Bernoulli over [reject, accept].
        def _vote_bias(key, shape, dtype=jnp.float32):
            return jnp.array([0.0, self.accept_bias], dtype=dtype)

        vote_logits = nn.Dense(2, kernel_init=orthogonal(0.01),
                               bias_init=_vote_bias)(h)
        pi_vote = distrax.Categorical(logits=vote_logits)

        critic = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(h)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        if self.aux_heads:
            # The two branches a vote chooses between, estimated separately rather
            # than differenced out of one critic: the critic averages over what the
            # OTHER voters did, so it cannot say what THIS vote changed. Both are
            # only meaningful on a vote-pass state (an offer is on the table), and
            # both read the same trunk as the vote head, so they see the theta the
            # vote is deciding on.
            def branch(name):
                z = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0), name=f"{name}_hidden")(h)
                z = act(z)
                z = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
                             name=f"{name}_out")(z)
                return jnp.squeeze(z, axis=-1)

            lock_value = branch("lock")     # this offer binds, now
            cont_value = branch("cont")     # it does not: null segment, reopen
            if return_aux:
                return pi_theta, pi_vote, critic, lock_value, cont_value
        elif return_aux:
            raise ValueError(
                "return_aux=True needs BargainingActorCritic(aux_heads=True): the "
                "branch value heads do not exist on this module. Build the network "
                "with aux_heads inferred from the checkpoint "
                "(bargain.params_have_aux_heads)."
            )

        return pi_theta, pi_vote, critic
