"""Formal contracting on Clean Up. Hydra entry point; the algorithm is moca_cnn.py.

The contract space is the paper's: theta = payment per waste cell cleaned, funded
evenly by the other agents (contracts.CleanupContract, theta in [0, 0.2]).

This is also the only environment carrying this repo's own extensions -- the
Rubinstein bargaining arm (TRAINING_MODE=joint), the harvest-tax contract kind, and
the claims/audits channel. Harvest and the Coin Game run the published arms only.
"""
from algorithms.MOCA.moca_cnn import make_train  # noqa: F401  (Hydra entry point)

SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_cleanup"}
TUNE_KWARGS = {"sweep_name": "cleanup"}
