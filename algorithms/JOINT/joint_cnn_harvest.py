"""Centralised joint control on harvest. Hydra entry point; the loop is joint_cnn.py."""
from algorithms.JOINT.joint_cnn import make_train  # noqa: F401  (Hydra entry point)

SINGLE_RUN_KWARGS = {"wandb_name": "joint_cnn_harvest"}
TUNE_KWARGS = {"sweep_name": "harvest"}
