from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import masked_mean, ensure_bool_mask


@dataclass
class FusionOut:
    e_final: torch.Tensor         # (B, D)
    beta: Optional[torch.Tensor]  # (B, 3) for GMF, else None
    e_text: torch.Tensor          # (B, D)
    e_audio: torch.Tensor         # (B, D)
    e_video: torch.Tensor         # (B, D)


class MHCA(nn.Module):
    """
    Cross-modal multi-head cross-attention:
      - Text queries attend to Audio tokens
      - Text queries attend to Video tokens
    computed within the same utterance u.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_ta = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_tv = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        t: torch.Tensor, t_mask: torch.Tensor,
        a: torch.Tensor, a_mask: torch.Tensor,
        v: torch.Tensor, v_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
          t: (B,L,D), a: (B,Ta,D), v: (B,Tv,D)
          *_mask: (B,*) float/bool with 1/True = valid

        Returns:
          t_aligned: (B,L,D) (text tokens aligned via cross-modal attention)
        """
        t_mask_b = ensure_bool_mask(t_mask)
        a_mask_b = ensure_bool_mask(a_mask)
        v_mask_b = ensure_bool_mask(v_mask)

        # key_padding_mask expects True for padding -> invert valid mask
        a_kpm = ~a_mask_b
        v_kpm = ~v_mask_b

        # Cross-attend: text queries
        ta, _ = self.attn_ta(query=t, key=a, value=a, key_padding_mask=a_kpm)
        tv, _ = self.attn_tv(query=t, key=v, value=v, key_padding_mask=v_kpm)

        # Residual + norm (stable alignment)
        y = (t + self.drop(ta) + self.drop(tv)) / 3.0
        y = self.norm(y)

        # Ensure padded text positions are zeroed (optional but helps pooling)
        y = y * t_mask_b.to(y.dtype).unsqueeze(-1)
        return y


class GMF(nn.Module):
    """
    Gated Multimodal Fusion producing beta(u) in R^3 and E_final(u).
    """
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, e_t: torch.Tensor, e_a: torch.Tensor, e_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e_*: (B, D)
        returns:
          e_final: (B,D)
          beta: (B,3) softmax weights
        """
        z = torch.cat([e_t, e_a, e_v], dim=-1)
        beta_logits = self.mlp(z)
        beta = torch.softmax(beta_logits, dim=-1)
        e_final = beta[:, 0:1] * e_t + beta[:, 1:2] * e_a + beta[:, 2:3] * e_v
        return e_final, beta


class FusionBlock(nn.Module):
    """
    Implements fusion modes:
      - mhca_gmf (full)
      - mhca_only
      - gmf_only
      - concat
    Plus modality toggles handled upstream by providing zeroed tokens/masks.
    """
    def __init__(
        self,
        d_model: int,
        fusion_mode: str = "mhca_gmf",
        mhca_heads: int = 4,
        gmf_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode

        self.mhca = MHCA(d_model=d_model, n_heads=mhca_heads, dropout=dropout)
        self.gmf = GMF(d_model=d_model, hidden=gmf_hidden, dropout=dropout)

        # For concat baseline and mhca_only combine
        self.concat_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        t_tok: torch.Tensor, t_mask: torch.Tensor,
        a_tok: torch.Tensor, a_mask: torch.Tensor,
        v_tok: torch.Tensor, v_mask: torch.Tensor,
    ) -> FusionOut:
        """
        Inputs are token sequences for a *single utterance* batch.
        Output is per-utterance fused embedding.
        """
        # pooled unimodal embeddings (static)
        e_t = masked_mean(t_tok, t_mask)  # (B,D)
        e_a = masked_mean(a_tok, a_mask)
        e_v = masked_mean(v_tok, v_mask)

        beta = None

        if self.fusion_mode == "mhca_gmf":
            t_aligned = self.mhca(t_tok, t_mask, a_tok, a_mask, v_tok, v_mask)
            e_t_al = masked_mean(t_aligned, t_mask)
            e_final, beta = self.gmf(e_t_al, e_a, e_v)
            return FusionOut(e_final=e_final, beta=beta, e_text=e_t_al, e_audio=e_a, e_video=e_v)

        if self.fusion_mode == "mhca_only":
            t_aligned = self.mhca(t_tok, t_mask, a_tok, a_mask, v_tok, v_mask)
            e_t_al = masked_mean(t_aligned, t_mask)
            # combine aligned text with static pooled A/V (no learned gating)
            e_final = self.concat_proj(torch.cat([e_t_al, e_a, e_v], dim=-1))
            return FusionOut(e_final=e_final, beta=None, e_text=e_t_al, e_audio=e_a, e_video=e_v)

        if self.fusion_mode == "gmf_only":
            # No MHCA; use static pooled features
            e_final, beta = self.gmf(e_t, e_a, e_v)
            return FusionOut(e_final=e_final, beta=beta, e_text=e_t, e_audio=e_a, e_video=e_v)

        if self.fusion_mode == "concat":
            e_final = self.concat_proj(torch.cat([e_t, e_a, e_v], dim=-1))
            return FusionOut(e_final=e_final, beta=None, e_text=e_t, e_audio=e_a, e_video=e_v)

        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
