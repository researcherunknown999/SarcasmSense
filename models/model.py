from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .common import ModalityToggles
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .fusion import FusionBlock


@dataclass
class ModelConfig:
    # Backbones
    text_model_name: str = "microsoft/deberta-v3-base"
    audio_model_name: str = "facebook/wav2vec2-base"
    video_model_name: str = "facebook/timesformer-base-finetuned-k400"

    # Dimensions
    d_model: int = 256
    dropout: float = 0.1

    # MHCA/GMF
    fusion_mode: str = "mhca_gmf"   # mhca_gmf | mhca_only | gmf_only | concat
    mhca_heads: int = 4
    gmf_hidden: int = 256

    # Audio encoder
    audio_cnn_channels: int = 256
    audio_cnn_kernel: int = 5
    audio_cnn_layers: int = 2
    audio_lstm_hidden: int = 256
    audio_lstm_layers: int = 1

    # OpenFace
    use_openface: bool = True
    openface_dim: int = 35

    # Tasks
    num_target_classes: int = 5  # set to match your manuscript

    # Modality toggles (for modality-removal ablations)
    use_text: bool = True
    use_audio: bool = True
    use_video: bool = True


class SarcasmSenseModel(nn.Module):
    """
    Utterance-level SarcasmSense model:
      - encodes transcript/audio/video within utterance u
      - applies MHCA + GMF *within the same utterance*
      - predicts sarcasm, sentiment shift, rhetorical target independently per utterance
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.text_enc = TextEncoder(cfg.text_model_name, d_model=cfg.d_model, dropout=cfg.dropout)
        self.audio_enc = AudioEncoder(
            cfg.audio_model_name,
            d_model=cfg.d_model,
            cnn_channels=cfg.audio_cnn_channels,
            cnn_kernel=cfg.audio_cnn_kernel,
            cnn_layers=cfg.audio_cnn_layers,
            lstm_hidden=cfg.audio_lstm_hidden,
            lstm_layers=cfg.audio_lstm_layers,
            dropout=cfg.dropout,
        )
        self.video_enc = VideoEncoder(
            cfg.video_model_name,
            d_model=cfg.d_model,
            use_openface=cfg.use_openface,
            openface_dim=cfg.openface_dim,
            dropout=cfg.dropout,
        )

        self.fusion = FusionBlock(
            d_model=cfg.d_model,
            fusion_mode=cfg.fusion_mode,
            mhca_heads=cfg.mhca_heads,
            gmf_hidden=cfg.gmf_hidden,
            dropout=cfg.dropout,
        )

        self.head_sarc = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 2),
        )
        self.head_shift = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 2),
        )
        self.head_target = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_target_classes),
        )

        self.toggles = ModalityToggles(cfg.use_text, cfg.use_audio, cfg.use_video)

    @staticmethod
    def _apply_toggle(tokens: torch.Tensor, mask: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled:
            return tokens, mask
        # Disable by zeroing mask and tokens
        z = torch.zeros_like(tokens)
        m = torch.zeros_like(mask)
        return z, m

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_wav: torch.Tensor,
        audio_mask: torch.Tensor,
        video_frames: torch.Tensor,
        video_mask: torch.Tensor,
        openface_au: Optional[torch.Tensor] = None,
        openface_cov: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with logits and (optionally) fusion diagnostics.
        """
        # Encode tokens
        t_tok = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)  # (B,L,D)
        t_mask = attention_mask.float()

        a_tok, a_mask = self.audio_enc(audio_wav=audio_wav, audio_mask=audio_mask)  # (B,Ta,D), (B,Ta)
        v_tok, v_mask = self.video_enc(
            video_frames=video_frames, video_mask=video_mask, openface_au=openface_au, openface_cov=openface_cov
        )  # (B,Tv,D), (B,Tv)

        # Apply modality-removal toggles
        t_tok, t_mask = self._apply_toggle(t_tok, t_mask, self.toggles.use_text)
        a_tok, a_mask = self._apply_toggle(a_tok, a_mask, self.toggles.use_audio)
        v_tok, v_mask = self._apply_toggle(v_tok, v_mask, self.toggles.use_video)

        # Fuse (utterance-level)
        fout = self.fusion(t_tok, t_mask, a_tok, a_mask, v_tok, v_mask)  # per-utterance embedding

        logits_sarc = self.head_sarc(fout.e_final)
        logits_shift = self.head_shift(fout.e_final)
        logits_target = self.head_target(fout.e_final)

        out: Dict[str, torch.Tensor] = {
            "logits_sarc": logits_sarc,
            "logits_shift": logits_shift,
            "logits_target": logits_target,
        }

        if return_aux:
            if fout.beta is not None:
                out["beta"] = fout.beta
            out["e_text"] = fout.e_text
            out["e_audio"] = fout.e_audio
            out["e_video"] = fout.e_video
            out["e_final"] = fout.e_final

        return out

    @classmethod
    def from_cfg_dict(cls, cfg: Dict[str, Any]) -> "SarcasmSenseModel":
        """
        Convenience: build ModelConfig from a nested config dict.
        Expects keys under cfg['model'].
        """
        m = cfg["model"]
        mc = ModelConfig(
            text_model_name=str(m["text_model_name"]),
            audio_model_name=str(m["audio_model_name"]),
            video_model_name=str(m["video_model_name"]),
            d_model=int(m.get("d_model", 256)),
            dropout=float(m.get("dropout", 0.1)),
            fusion_mode=str(m.get("fusion_mode", "mhca_gmf")),
            mhca_heads=int(m.get("mhca_heads", 4)),
            gmf_hidden=int(m.get("gmf_hidden", 256)),
            audio_cnn_channels=int(m.get("audio_cnn_channels", 256)),
            audio_cnn_kernel=int(m.get("audio_cnn_kernel", 5)),
            audio_cnn_layers=int(m.get("audio_cnn_layers", 2)),
            audio_lstm_hidden=int(m.get("audio_lstm_hidden", 256)),
            audio_lstm_layers=int(m.get("audio_lstm_layers", 1)),
            use_openface=bool(m.get("use_openface", True)),
            openface_dim=int(m.get("openface_dim", 35)),
            num_target_classes=int(m.get("num_target_classes", 5)),
            use_text=bool(m.get("use_text", True)),
            use_audio=bool(m.get("use_audio", True)),
            use_video=bool(m.get("use_video", True)),
        )
        return cls(mc)
