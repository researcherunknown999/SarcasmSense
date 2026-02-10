from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Best-effort reproducibility across Python/NumPy/PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some environments may not support full deterministic algorithms
            pass


def worker_init_fn(worker_id: int) -> None:
    """Deterministic DataLoader worker seeding."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
