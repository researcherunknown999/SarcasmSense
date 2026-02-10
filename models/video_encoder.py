from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import TimesformerModel
except Exception:  # pragma: no cover
    TimesformerModel = None  # type: ignore


class VideoEncoder(nn.Module):
    """
    TimeSformer -> token sequence -> project to d_model.
    Optionally append OpenFace AU token (MLP) as extra token.
    """
    def __init__(
        self,
        model_name: str,
        d_model: int,
        use_openface: bool = True,
        openface_dim: int = 35,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if TimesformerModel is None:
            raise ImportError("TimesformerModel is not available. Please install a transformers version with TimeSformer support.")
        self.backbone = TimesformerModel.from_pretrained(model_name)
        h = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(h, d_model)
        self.drop = nn.Dropout(dropout)

        self.use_openface = bool(use_openface)
        self.openface_dim = int(openface_dim)
        if self.use_openface:
            self.au_mlp = nn.Sequential(
                nn.Linear(self.openface_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

    def forward(
        self,
        video_frames: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        openface_au: Optional[torch.Tensor] = None,
        openface_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        video_frames: (B, T, 3, H, W) uint8/float (we convert to float)
        video_mask: (B, T) with 1=valid sampled, 0=pad (best-effort)
        openface_au: (B, F)
        openface_cov: (B,) 1 if available else 0

        returns:
          tokens: (B, Tv, D)  (Tv includes CLS + patch tokens + optional AU token)
          token_mask: (B, Tv) float mask
        """
        # transformers expects float pixel_values
        if video_frames.dtype != torch.float32:
            pixel_values = video_frames.float() / 255.0
        else:
            pixel_values = video_frames

        out = self.backbone(pixel_values=pixel_values)
        x = out.last_hidden_state  # (B, Tv, H)
        x = self.drop(x)
        x = self.proj(x)           # (B, Tv, D)

        B, Tv, _ = x.shape
        tok_mask = torch.ones((B, Tv), device=x.device, dtype=torch.float32)

        # Optionally append AU token
        if self.use_openface and (openface_au is not None) and (openface_cov is not None):
            au_tok = self.au_mlp(openface_au.float())  # (B,D)
            au_tok = au_tok.unsqueeze(1)               # (B,1,D)

            # coverage mask: if cov=0, keep token but mark as padded
            cov = openface_cov.float().clamp(0.0, 1.0).to(x.device)  # (B,)
            x = torch.cat([x, au_tok], dim=1)                         # (B,Tv+1,D)
            au_mask = cov.unsqueeze(1)                                # (B,1)
            tok_mask = torch.cat([tok_mask, au_mask], dim=1)          # (B,Tv+1)

        return x, tok_mask
