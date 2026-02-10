from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summaries(root: Path) -> List[Path]:
    return sorted(root.rglob("summary.json"))


def pick_best_seed(summaries: List[Path]) -> Dict[str, Path]:
    """
    Pick best seed per variant using best_val_score in summary.json.
    Expects .../<variant>/seed_k/summary.json.
    """
    best: Dict[str, Tuple[float, Path]] = {}
    for p in summaries:
        s = load_json(p)
        try:
            variant = p.parent.parent.name
        except Exception:
            variant = "unknown"
        score = float(s.get("best_val_score", float("-inf")))
        if (variant not in best) or (score > best[variant][0]):
            best[variant] = (score, p)
    return {k: v[1] for k, v in best.items()}


def plot_cm(cm: np.ndarray, title: str, out_path: Path, class_names: List[str] | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts/figs")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    summaries = find_summaries(runs_root)
    if not summaries:
        raise SystemExit(f"No summary.json found under {runs_root}")

    best = pick_best_seed(summaries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant, spath in best.items():
        s = load_json(spath)
        split = s[args.split]
        cms = split.get("confusions", {})
        if not cms:
            print(f"[WARN] No confusions found for {variant} at {spath}")
            continue

        cm_sarc = np.array(cms["sarc"], dtype=int)
        cm_shift = np.array(cms["shift"], dtype=int)
        cm_target = np.array(cms["target"], dtype=int)

        plot_cm(cm_sarc, f"{variant} — Sarcasm CM ({args.split})", out_dir / variant / f"cm_sarcasm_{args.split}.png", ["0", "1"])
        plot_cm(cm_shift, f"{variant} — Shift CM ({args.split})", out_dir / variant / f"cm_shift_{args.split}.png", ["0", "1"])

        # target labels unknown; use indices
        target_names = [str(i) for i in range(cm_target.shape[0])]
        plot_cm(cm_target, f"{variant} — Target CM ({args.split})", out_dir / variant / f"cm_target_{args.split}.png", target_names)

    print(f"Wrote confusion matrix figures to: {out_dir}")


if __name__ == "__main__":
    main()
