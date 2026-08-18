"""The centralised joint controller: one network choosing every agent's action.

This is the network behind the welfare-maximising upper bound, and it is
deliberately the ONLY thing about that baseline which differs from ordinary PPO.
Everything else -- the trunk, the optimiser, the losses -- is shared with the
decentralised arms, so a gap between them is attributable to the control structure
rather than to architecture or tuning.
"""
from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.utils.networks import CNN


class CentralisedActorCritic(nn.Module):
    """One actor-critic over the JOINT action, given every agent's observation.

    The reference implementation (JointEnv, two_stage_train.py) builds this as an
    environment wrapper: a single nominal agent `a0` receives the concatenated
    observations, emits a vector that is sliced back into per-agent actions, and is
    paid the summed reward. The same construction is expressed here as a network,
    which is the natural form in a vectorised JAX loop -- there is no RLlib
    multi-agent bookkeeping to satisfy, so nothing is gained by pretending N agents
    exist when only one policy does.

    ACTION HEADS, NOT A JOINT CATEGORICAL. A fully joint distribution over
    |A|^N outcomes is 9^7 = 4.8M logits on Clean Up and cannot be represented. This
    emits N independent categoricals conditioned on the SHARED state, so

        pi(a_1..a_N | s) = prod_i pi_i(a_i | s),

    which is exactly what the reference's Box-of-N-actions gives, and what
    `acts['a0'][i]` slices out of it. The restriction is that the agents cannot
    correlate their actions beyond what the common state already tells them --
    they cannot flip a shared coin to break a symmetry. That costs nothing here:
    the environments have no coordination problem that requires correlated
    randomisation, and the state is fully shared, so any deterministic optimal
    joint policy remains representable.

    The critic is a single head. It values the joint state under the summed reward,
    which is the object this baseline maximises and the reason it has no credit
    assignment problem: there is one return, one gradient, and no question of which
    agent earned what.

    Attributes:
        num_agents: N, the number of action heads.
        action_dim: |A|, the size of one agent's action set.
        activation: "relu" or "tanh".
    """
    num_agents: int
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        embedding = CNN(self.activation)(x)

        actor = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor = activation(actor)
        # One block of logits per agent, produced by one head and then split. A head
        # per agent would be equivalent; this keeps the parameter tree flat and the
        # split explicit.
        logits = nn.Dense(
            self.num_agents * self.action_dim,
            kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(actor)
        logits = logits.reshape(*logits.shape[:-1], self.num_agents, self.action_dim)
        pi = distrax.Categorical(logits=logits)      # batch (..., N), event ()

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


def joint_log_prob(pi, actions):
    """Log-probability of the JOINT action: the sum over the per-agent heads.

    The factored policy means the joint log-prob is a sum, and PPO's ratio has to be
    computed on that joint quantity -- one action was taken (a vector), and one
    importance weight belongs to it. Summing here rather than at each call site is
    what keeps the ratio from being silently computed per agent, which would apply
    the clip N times to N different ratios and quietly stop being PPO.
    """
    return pi.log_prob(actions).sum(axis=-1)


def joint_entropy(pi):
    """Entropy of the joint policy: the sum of the heads', by independence."""
    return pi.entropy().sum(axis=-1)
