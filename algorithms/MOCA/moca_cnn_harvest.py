"""Formal contracting on Harvest. Hydra entry point; the algorithm is moca_cnn.py.

The contract space is the authors' `HarvestFeaturemodLocalContract`: an agent that
eats an apple in a low-density patch pays theta to the other agents, split evenly,
with theta in [0, 10] (contracts.HarvestContract).

Note the direction. Clean Up's contract SUBSIDISES an under-provided public good;
this one TAXES an over-used common resource. Running both is the point -- they are
the two halves of Ostrom's provision/appropriation problem, and a mechanism that
handles one need not handle the other.
"""
from algorithms.MOCA.moca_cnn import make_train  # noqa: F401  (Hydra entry point)

SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_harvest"}
TUNE_KWARGS = {"sweep_name": "harvest"}
