from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch

from sarcasmsense.utils.config import load_yaml, deep_update, set_by_dotted_path
from sarcasmsense.utils.repro import set_global_seed
from sarcasmsense.utils.io import ensure_dir, save_json
from sarcasmsense.data.loaders import build_loaders
from sarcasmsense.models import SarcasmSenseModel
from sarcasmsense.training import Trainer


def _parse_override_kv(kv: str) -> tuple[str, Any]:
    """
    Parse dotted override like:
      model.fusion_mode=mhca_gmf
      train.lr_text=3e-5
      model.use_audio=false
    Value is parsed via yaml.safe_load for correct types.
    """
    if "=" not in kv:
        raise ValueError(f"Override must be key=value, got: {kv}")
    k, v = kv.split("=", 1)
    k = k.strip()
    v_parsed = yaml.safe_load(v)
    return k, v_parsed


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    cfg = dict(cfg)
    for kv in overrides:
        k, v = _parse_override_kv(kv)
        set_by_dotted_path(cfg, k, v)
    return cfg


def pick_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to base YAML config (e.g., configs/default.yaml)")
    ap.add_argument("--data_root", type=str, required=True, help="Dataset root containing manifest.csv and utterance files")
    ap.add_argument("--run_dir", type=str, required=True, help="Output directory for this run (artifacts/runs/...)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0")
    ap.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Dotted overrides: key=value (e.g., model.use_audio=false train.batch_size=8)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    # Reproducibility
    set_global_seed(int(args.seed), deterministic=bool(cfg.get("repro", {}).get("deterministic", True)))

    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    # Save effective config for reproducibility
    save_json(run_dir / "config_effective.json", cfg)

    device = pick_device(args.device)

    # Data
    loaders = build_loaders(cfg, data_root=args.data_root, seed=int(args.seed))

    # Model
    model = SarcasmSenseModel.from_cfg_dict(cfg)

    # Train/Eval
    trainer = Trainer(cfg=cfg, model=model, device=device, run_dir=run_dir, seed=int(args.seed))
    summary = trainer.fit(loaders)

    # Save top-level summary (already written in Trainer, but keep here too)
    save_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
