from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, T, D)
    mask: (B, T) with 1 for valid, 0 for pad
    returns: (B, D)
    """
    mask = mask.to(dtype=x.dtype)
    mask = mask.unsqueeze(-1)  # (B,T,1)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den


def ensure_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert {0,1} float mask to bool mask (True=valid)."""
    if mask.dtype == torch.bool:
        return mask
    return mask > 0.5


@dataclass
class ModalityToggles:
    use_text: bool = True
    use_audio: bool = True
    use_video: bool = True
