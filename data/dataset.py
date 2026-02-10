from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import soundfile as sf
import cv2

from transformers import AutoTokenizer


def _read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def _resample_linear(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Deterministic linear resampling (no external deps)."""
    if sr_in == sr_out:
        return wav.astype(np.float32)
    n_out = int(round(len(wav) * (sr_out / sr_in)))
    if n_out <= 1:
        return np.zeros((sr_out,), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _read_audio(path: str, target_sr: int, max_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    max_len = int(target_sr * max_sec)

    if not p.exists():
        return np.zeros((max_len,), dtype=np.float32), np.zeros((max_len,), dtype=np.float32)

    wav, sr = sf.read(str(p))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)
    wav = _resample_linear(wav, sr, target_sr)
    orig_len = min(len(wav), max_len)

    wav = wav[:max_len]
    if len(wav) < max_len:
        wav = np.pad(wav, (0, max_len - len(wav)))

    mask = np.zeros((max_len,), dtype=np.float32)
    mask[:orig_len] = 1.0
    return wav, mask


def _sample_video_frames(path: str, num_frames: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample frames across the entire utterance clip.
    Returns:
      frames: (T, H, W, 3) uint8 RGB
      mask:   (T,) float32, 1 for real sampled frames, 0 for padded
    """
    p = Path(path)
    frames = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    mask = np.zeros((num_frames,), dtype=np.float32)

    if not p.exists():
        return frames, mask

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return frames, mask

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # Fallback: try to read sequentially and store what we can
        collected = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            collected.append(frame)
            if len(collected) >= num_frames:
                break
        cap.release()

        if len(collected) == 0:
            return frames, mask

        # Resize/convert and pad
        for i in range(min(num_frames, len(collected))):
            fr = cv2.cvtColor(collected[i], cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
            frames[i] = fr
            mask[i] = 1.0
        # If not enough, pad using last real frame
        last = frames[int(mask.sum()) - 1] if mask.sum() > 0 else np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(int(mask.sum()), num_frames):
            frames[i] = last
            mask[i] = 0.0
        return frames, mask

    idxs = np.linspace(0, max(0, total - 1), num=num_frames).astype(int).tolist()
    wanted = set(idxs)

    # Read once, collect selected indices
    i = 0
    j = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in wanted:
            fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
            frames[j] = fr
            mask[j] = 1.0
            j += 1
            if j >= num_frames:
                break
        i += 1

    cap.release()

    if j == 0:
        return frames, mask

    # pad remaining with last real frame
    last = frames[j - 1].copy()
    for k in range(j, num_frames):
        frames[k] = last
        mask[k] = 0.0

    return frames, mask


def _read_openface_au(path: str, expected_dim: int) -> Tuple[np.ndarray, float]:
    """
    Read OpenFace AU CSV and mean-pool AU columns.
    Returns:
      au_vec: (expected_dim,) float32
      cov: 1.0 if present, else 0.0
    """
    p = Path(path) if path else None
    if (p is None) or (not p.exists()):
        return np.zeros((expected_dim,), dtype=np.float32), 0.0

    df = pd.read_csv(str(p))
    # Prefer AU columns: AU01_r, AU12_c, etc.
    au_cols = [c for c in df.columns if c.lower().startswith("au")]
    if not au_cols:
        # Fallback: numeric columns only
        au_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not au_cols:
        return np.zeros((expected_dim,), dtype=np.float32), 0.0

    x = df[au_cols].to_numpy(dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((expected_dim,), dtype=np.float32), 0.0

    vec = x.mean(axis=0)
    if vec.shape[0] < expected_dim:
        vec = np.pad(vec, (0, expected_dim - vec.shape[0]))
    vec = vec[:expected_dim].astype(np.float32)
    return vec, 1.0


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    audio_wav: torch.Tensor
    audio_mask: torch.Tensor
    video_frames: torch.Tensor
    video_mask: torch.Tensor
    openface_au: torch.Tensor
    openface_cov: torch.Tensor
    y_sarc: torch.Tensor
    y_shift: torch.Tensor
    y_target: torch.Tensor
    utt_id: list[str]


class SarcasmSenseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str,
        max_text_len: int,
        audio_sr: int,
        audio_max_sec: float,
        video_num_frames: int,
        video_size: int,
        use_openface: bool,
        openface_dim: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_text_len = int(max_text_len)
        self.audio_sr = int(audio_sr)
        self.audio_max_sec = float(audio_max_sec)
        self.video_num_frames = int(video_num_frames)
        self.video_size = int(video_size)
        self.use_openface = bool(use_openface)
        self.openface_dim = int(openface_dim)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]

        # Text
        text = _read_text(str(r["transcript_path"]))
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Audio
        wav, amask = _read_audio(str(r["audio_path"]), self.audio_sr, self.audio_max_sec)

        # Video
        frames, vmask = _sample_video_frames(str(r["video_path"]), self.video_num_frames, self.video_size)

        # OpenFace AU
        if self.use_openface:
            au, cov = _read_openface_au(str(r.get("openface_path", "")), self.openface_dim)
        else:
            au, cov = np.zeros((self.openface_dim,), dtype=np.float32), 0.0

        return {
            "utt_id": str(r["utt_id"]),
            "input_ids": tok["input_ids"].squeeze(0),          # (L,)
            "attention_mask": tok["attention_mask"].squeeze(0),# (L,)
            "audio_wav": torch.from_numpy(wav),                # (Ta,)
            "audio_mask": torch.from_numpy(amask),             # (Ta,)
            "video_frames": torch.from_numpy(frames).permute(0, 3, 1, 2),  # (T,3,H,W)
            "video_mask": torch.from_numpy(vmask),             # (T,)
            "openface_au": torch.from_numpy(au),               # (F,)
            "openface_cov": torch.tensor(cov, dtype=torch.float32),
            "y_sarc": torch.tensor(int(r["y_sarc"]), dtype=torch.long),
            "y_shift": torch.tensor(int(r["y_shift"]), dtype=torch.long),
            "y_target": torch.tensor(int(r["y_target"]), dtype=torch.long),
        }


def collate_fn(items: list[Dict[str, Any]]) -> Batch:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([it[key] for it in items], dim=0)

    return Batch(
        input_ids=stack("input_ids"),
        attention_mask=stack("attention_mask"),
        audio_wav=stack("audio_wav"),
        audio_mask=stack("audio_mask"),
        video_frames=stack("video_frames"),
        video_mask=stack("video_mask"),
        openface_au=stack("openface_au"),
        openface_cov=torch.stack([it["openface_cov"] for it in items], dim=0),
        y_sarc=stack("y_sarc"),
        y_shift=stack("y_shift"),
        y_target=stack("y_target"),
        utt_id=[it["utt_id"] for it in items],
    )
