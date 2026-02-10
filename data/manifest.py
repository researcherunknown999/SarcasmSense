from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd


REQUIRED_COLS: List[str] = [
    "utt_id",
    "transcript_path",
    "audio_path",
    "video_path",
    "y_sarc",
    "y_shift",
    "y_target",
]

OPTIONAL_COLS: List[str] = ["split", "openface_path"]


def load_manifest(data_root: str | Path, manifest_rel: str = "manifest.csv") -> pd.DataFrame:
    data_root = Path(data_root)
    mpath = data_root / manifest_rel
    if not mpath.exists():
        raise FileNotFoundError(f"manifest not found: {mpath}")

    df = pd.read_csv(mpath)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"manifest missing required column: {c}")

    # Ensure optional columns exist
    if "split" not in df.columns:
        df["split"] = ""
    if "openface_path" not in df.columns:
        df["openface_path"] = ""

    return df


def resolve_paths(df: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    """Resolve relative paths in manifest to absolute paths."""
    data_root = Path(data_root).resolve()
    df = df.copy()

    def _resolve(p: str) -> str:
        p = str(p) if p is not None else ""
        if p.strip() == "" or p.lower() == "nan":
            return ""
        return str((data_root / p).resolve())

    for col in ["transcript_path", "audio_path", "video_path", "openface_path"]:
        df[col] = df[col].apply(_resolve)

    return df
