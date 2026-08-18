"""Formal contracting on the Coin Game. Hydra entry point; the algorithm is moca_cnn.py.

The contract space is theta per coin taken of another agent's colour, paid to the
agent whose colour it was, with theta in [0, 2] (contracts.CoinGameContract).

THE AUTHORS DEFINE NO COIN GAME CONTRACT -- their release covers Cleanup, Harvest and
a self-driving domain. This space is built from their design rules rather than
transcribed from their code, and the class docstring in contracts.py sets out which
rules and why the range is what it is. Report it as such.

Two agents, not seven: the environment is red-vs-green by construction. That is the
smallest setting the paper reports (it runs 2, 4 and 8 agents), and it is the one
where the contract is cleanest -- with a single counterparty there is no question of
who bore the cost of a stolen coin.
"""
from algorithms.MOCA.moca_cnn import make_train  # noqa: F401  (Hydra entry point)

SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_coins"}
TUNE_KWARGS = {"sweep_name": "coins"}
