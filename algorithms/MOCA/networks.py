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
