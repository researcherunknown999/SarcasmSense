from __future__ import annotations

from typing import Any, Dict
from torch.utils.data import DataLoader
import pandas as pd

from .manifest import load_manifest, resolve_paths
from .split import stratified_split_if_missing
from .dataset import SarcasmSenseDataset, collate_fn
from ..utils.repro import worker_init_fn


def build_loaders(cfg: Dict[str, Any], data_root: str, seed: int) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from config and a local manifest.csv.
    Expects cfg structure (keys used in this chunk):
      cfg['data'].{manifest,num_workers,max_text_len,audio_sr,audio_max_sec,video_num_frames,video_size,stratify_split_if_missing,split{train,val,test}}
      cfg['model'].{text_model_name,use_openface,openface_dim}
      cfg['train'].batch_size
    """
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    df = load_manifest(data_root, dcfg.get("manifest", "manifest.csv"))
    df = resolve_paths(df, data_root)

    if bool(dcfg.get("stratify_split_if_missing", True)):
        df = stratified_split_if_missing(
            df,
            train_frac=float(dcfg["split"]["train"]),
            val_frac=float(dcfg["split"]["val"]),
            test_frac=float(dcfg["split"]["test"]),
            seed=seed,
        )

    df_train = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    df_val = df[df["split"].astype(str) == "val"].reset_index(drop=True)
    df_test = df[df["split"].astype(str) == "test"].reset_index(drop=True)

    tokenizer_name = str(mcfg["text_model_name"])
    use_openface = bool(mcfg.get("use_openface", True))
    openface_dim = int(mcfg.get("openface_dim", 35))

    train_ds = SarcasmSenseDataset(
        df_train,
        tokenizer_name=tokenizer_name,
        max_text_len=int(dcfg.get("max_text_len", 128)),
        audio_sr=int(dcfg.get("audio_sr", 16000)),
        audio_max_sec=float(dcfg.get("audio_max_sec", 10.0)),
        video_num_frames=int(dcfg.get("video_num_frames", 32)),
        video_size=int(dcfg.get("video_size", 224)),
        use_openface=use_openface,
        openface_dim=openface_dim,
    )
    val_ds = SarcasmSenseDataset(
        df_val,
        tokenizer_name=tokenizer_name,
        max_text_len=int(dcfg.get("max_text_len", 128)),
        audio_sr=int(dcfg.get("audio_sr", 16000)),
        audio_max_sec=float(dcfg.get("audio_max_sec", 10.0)),
        video_num_frames=int(dcfg.get("video_num_frames", 32)),
        video_size=int(dcfg.get("video_size", 224)),
        use_openface=use_openface,
        openface_dim=openface_dim,
    )
    test_ds = SarcasmSenseDataset(
        df_test,
        tokenizer_name=tokenizer_name,
        max_text_len=int(dcfg.get("max_text_len", 128)),
        audio_sr=int(dcfg.get("audio_sr", 16000)),
        audio_max_sec=float(dcfg.get("audio_max_sec", 10.0)),
        video_num_frames=int(dcfg.get("video_num_frames", 32)),
        video_size=int(dcfg.get("video_size", 224)),
        use_openface=use_openface,
        openface_dim=openface_dim,
    )

    batch_size = int(tcfg.get("batch_size", 8))
    num_workers = int(dcfg.get("num_workers", 4))

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        ),
    }
    return loaders
