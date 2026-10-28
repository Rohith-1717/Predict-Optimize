"""
src/utils.py
Helper functions, random seed initialization, and utilities.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CPU device for fast, lightweight laptop execution."""
    return torch.device("cpu")
