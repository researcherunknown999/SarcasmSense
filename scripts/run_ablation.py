from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import subprocess
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one(
    base_config: str,
    data_root: str,
    out_root: Path,
    variant_name: str,
    seed: int,
    overrides: Dict[str, Any],
    device: str,
) -> None:
    run_dir = out_root / variant_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    override_args: List[str] = []
    for k, v in overrides.items():
        override_args.append(f"{k}={v}")

    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        base_config,
        "--data_root",
        data_root,
        "--run_dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if override_args:
        cmd += ["--override"] + override_args

    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_config", type=str, default="configs/experiments_table7.yaml")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, default="artifacts/runs_ablation")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    exp = load_yaml(args.exp_config)
    base_config = exp["base_config"]
    variants = exp["variants"]
    seeds = args.seeds if args.seeds is not None else exp.get("seeds", list(range(1, 11)))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for var in variants:
        name = var["name"]
        overrides = var.get("overrides", {})
        for seed in seeds:
            run_one(
                base_config=base_config,
                data_root=args.data_root,
                out_root=out_root,
                variant_name=name,
                seed=int(seed),
                overrides=overrides,
                device=args.device,
            )


if __name__ == "__main__":
    main()
