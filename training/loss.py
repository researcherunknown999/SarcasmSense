from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class LossWeights:
    sarc: float = 0.60
    shift: float = 0.25
    target: float = 0.15


class MultiTaskLoss(nn.Module):
    """
    Weighted multitask cross-entropy loss for:
      - sarcasm (binary)
      - sentiment shift (binary)
      - target (K-way)
    """
    def __init__(self, w: LossWeights) -> None:
        super().__init__()
        self.w = w
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        y_sarc: torch.Tensor,
        y_shift: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ls = self.ce(logits["logits_sarc"], y_sarc)
        lsh = self.ce(logits["logits_shift"], y_shift)
        lt = self.ce(logits["logits_target"], y_target)

        total = self.w.sarc * ls + self.w.shift * lsh + self.w.target * lt
        return {"total": total, "sarc": ls, "shift": lsh, "target": lt}
