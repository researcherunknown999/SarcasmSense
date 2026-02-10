from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    DeBERTa (or any HF text model) -> token embeddings -> project to d_model.
    """
    def __init__(self, model_name: str, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        attention_mask: (B, L) with 1=valid
        return: (B, L, D)
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # (B,L,H)
        x = self.drop(x)
        x = self.proj(x)           # (B,L,D)
        return x
