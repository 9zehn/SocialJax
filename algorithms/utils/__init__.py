"""Shared utilities for all algorithms."""

from algorithms.utils.networks import (
    CNN,
    ActorCritic,
    Actor,
    Critic,
    SmallCNN,
    SmallActor,
    SmallCritic
)

from algorithms.utils.data_utils import (
    batchify,
    batchify_dict,
    batchify_numpy,
    unbatchify
)

from algorithms.utils.vdn_networks import (
    QNetwork
)

from algorithms.utils.io_utils import (
    save_params,
    load_params,
    checkpoint_filename,
    save_train_state,
    load_train_state,
    save_run_config,
    load_run_config,
    run_config_path,
    run_stem,
    contract_range,
    RUN_CONFIG_EXT
)

from algorithms.utils.eval_utils import (
    evaluate_ippo,
    evaluate_mappo_style
)

from algorithms.utils.types import (
    Transition,
    MAPPOTransition,
    IRATTransition,
)

from algorithms.utils.transfer_utils import (
    s_from_ratio,
)

__all__ = [
    # Network architectures
    "CNN",
    "ActorCritic",
    "Actor",
    "Critic",
    "SmallCNN",
    "SmallActor",
    "SmallCritic",
    # VDN-specific networks
    "QNetwork",
    # Data manipulation utilities
    "batchify",
    "batchify_dict",
    "batchify_numpy",
    "unbatchify",
    # IO utilities
    "save_params",
    "load_params",
    "checkpoint_filename",
    "save_train_state",
    "load_train_state",
    "save_run_config",
    "load_run_config",
    "run_config_path",
    "run_stem",
    "contract_range",
    "RUN_CONFIG_EXT",
    # Evaluation utilities
    "evaluate_ippo",
    "evaluate_mappo_style",
    # Shared types
    "Transition",
    "MAPPOTransition",
    "IRATTransition",
    # TRANSFER utilities
    "s_from_ratio",
]
