from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Wav2Vec2 -> temporal CNN -> BiLSTM -> project to d_model.
    Returns token sequence (B, Ta, D).
    """
    def __init__(
        self,
        model_name: str,
        d_model: int,
        cnn_channels: int = 256,
        cnn_kernel: int = 5,
        cnn_layers: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        h = int(self.backbone.config.hidden_size)

        convs = []
        in_ch = h
        for _ in range(int(cnn_layers)):
            convs.append(nn.Conv1d(in_ch, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2))
            convs.append(nn.GELU())
            convs.append(nn.Dropout(dropout))
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*convs)

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * lstm_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
        # mask: (B, T) float/bool, valid=1
        if mask.dtype == torch.bool:
            lengths = mask.long().sum(dim=1)
        else:
            lengths = (mask > 0.5).long().sum(dim=1)
        return lengths.clamp_min(1)

    def forward(self, audio_wav: torch.Tensor, audio_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio_wav: (B, Traw) float32
        audio_mask: (B, Traw) with 1=valid
        returns:
          tokens: (B, Ta, D)  (Ta is wav2vec2 time steps)
          token_mask: (B, Ta) float mask aligned to tokens
        """
        if audio_mask is None:
            attn_mask = None
        else:
            attn_mask = (audio_mask > 0.5).long()

        out = self.backbone(input_values=audio_wav, attention_mask=attn_mask)
        x = out.last_hidden_state  # (B, Ta, H)

        # Build token mask: wav2vec2 downsamples time; best-effort mask using output length
        B, Ta, _ = x.shape
        if audio_mask is None:
            tok_mask = torch.ones((B, Ta), device=x.device, dtype=torch.float32)
        else:
            # Approximate: assume tok length proportional to raw valid length
            raw_len = self._lengths_from_mask(audio_mask).to(x.device)  # (B,)
            # map raw_len -> tok_len by ratio
            tok_len = torch.clamp((raw_len.float() / audio_wav.shape[1]) * Ta, min=1.0, max=float(Ta)).long()
            tok_mask = torch.zeros((B, Ta), device=x.device, dtype=torch.float32)
            for b in range(B):
                tok_mask[b, : tok_len[b].item()] = 1.0

        # CNN over time: (B, H, Ta) -> (B, C, Ta)
        y = x.transpose(1, 2)
        y = self.cnn(y)
        y = y.transpose(1, 2)  # (B, Ta, C)

        # BiLSTM with packing using tok_mask lengths
        lengths = (tok_mask > 0.5).long().sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        y2, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=Ta)  # (B,Ta,2H)

        y2 = self.drop(y2)
        z = self.proj(y2)  # (B,Ta,D)
        return z, tok_mask
