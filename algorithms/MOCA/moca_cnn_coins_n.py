"""Formal contracting on the N-player Coin Game. Hydra entry point; the algorithm is
moca_cnn.py.

Seven agents, one coin colour each: taking a coin pays +1, and taking somebody
else's costs its owner -2. The contract prices exactly that theft, at theta per coin,
paid to the agent whose coin it was (contracts.CoinGameContract, unchanged from the
two-player environment -- it was written to work at any N).

Not a replacement for `--env coins`. With two agents that game is literal Rubinstein
alternating offers, and it stays registered for exactly that reason. What seven
agents buy is a real quorum, a real unanimity constraint, and a dilemma made harsher
by arithmetic: only one coin in seven is yours, so cooperating means walking past six
you may not take.
"""
from algorithms.MOCA.moca_cnn import make_train  # noqa: F401  (Hydra entry point)

SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_coins_n"}
TUNE_KWARGS = {"sweep_name": "coins_n"}
