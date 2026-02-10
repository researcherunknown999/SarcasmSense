from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summaries(root: Path) -> List[Path]:
    return sorted(root.rglob("summary.json"))


def mean_std_ci95(xs: List[float]) -> Tuple[float, float, float]:
    """
    Seed-level CI95 over runs: mean ± 1.96 * std/sqrt(n).
    Returns (mean, std, ci95_halfwidth). If n<2 -> std=0, ci=0.
    """
    x = np.array([float(v) for v in xs if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(x))
    if len(x) < 2:
        return mu, 0.0, 0.0
    sd = float(np.std(x, ddof=1))
    ci = 1.96 * sd / np.sqrt(len(x))
    return mu, sd, float(ci)


def fmt_mean_ci(mu: float, ci: float, scale: float = 100.0) -> str:
    if not np.isfinite(mu) or not np.isfinite(ci):
        return ""
    return f"{mu*scale:.2f} ± {ci*scale:.2f}"


@dataclass
class Extracted:
    variant: str
    seed: int
    best_val_score: float

    # test metrics (primary for tables)
    sarc_f1: float
    sarc_acc: float
    shift_f1: float
    shift_acc: float
    target_f1: float
    target_acc: float


def extract_one(summary_path: Path) -> Extracted:
    s = load_json(summary_path)

    # variant name = parent directory under the run root
    # .../<variant>/seed_k/summary.json
    try:
        variant = summary_path.parent.parent.name
    except Exception:
        variant = "unknown"

    seed = int(s.get("seed", -1))
    best_val_score = float(s.get("best_val_score", float("nan")))

    test = s["test"]
    sarc = test["sarc"]
    shift = test["shift"]
    target = test["target"]

    return Extracted(
        variant=variant,
        seed=seed,
        best_val_score=best_val_score,
        sarc_f1=float(sarc.get("macro_f1", float("nan"))),
        sarc_acc=float(sarc.get("acc", float("nan"))),
        shift_f1=float(shift.get("macro_f1", float("nan"))),
        shift_acc=float(shift.get("acc", float("nan"))),
        target_f1=float(target.get("macro_f1", float("nan"))),
        target_acc=float(target.get("acc", float("nan"))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="Root folder containing variant/seed_*/summary.json")
    ap.add_argument("--out_csv", type=str, default="artifacts/tables/aggregate.csv")
    ap.add_argument("--out_table_csv", type=str, default="artifacts/tables/table_like.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    summaries = find_summaries(runs_root)
    if not summaries:
        raise SystemExit(f"No summary.json found under {runs_root}")

    rows: List[Extracted] = [extract_one(p) for p in summaries]
    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["variant", "seed"]).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Aggregate per variant (mean ± CI95 over seeds)
    agg_rows = []
    for variant, g in df.groupby("variant"):
        mu_s, sd_s, ci_s = mean_std_ci95(g["sarc_f1"].tolist())
        mu_sa, sd_sa, ci_sa = mean_std_ci95(g["sarc_acc"].tolist())
        mu_sh, sd_sh, ci_sh = mean_std_ci95(g["shift_f1"].tolist())
        mu_sha, sd_sha, ci_sha = mean_std_ci95(g["shift_acc"].tolist())
        mu_t, sd_t, ci_t = mean_std_ci95(g["target_f1"].tolist())
        mu_ta, sd_ta, ci_ta = mean_std_ci95(g["target_acc"].tolist())

        agg_rows.append(
            {
                "variant": variant,
                "n_seeds": int(len(g)),
                "sarcasm_f1_mean": mu_s,
                "sarcasm_f1_std": sd_s,
                "sarcasm_f1_ci95": ci_s,
                "sarcasm_f1_meanpmci": fmt_mean_ci(mu_s, ci_s),

                "sarcasm_acc_mean": mu_sa,
                "sarcasm_acc_std": sd_sa,
                "sarcasm_acc_ci95": ci_sa,
                "sarcasm_acc_meanpmci": fmt_mean_ci(mu_sa, ci_sa),

                "shift_f1_mean": mu_sh,
                "shift_f1_std": sd_sh,
                "shift_f1_ci95": ci_sh,
                "shift_f1_meanpmci": fmt_mean_ci(mu_sh, ci_sh),

                "shift_acc_mean": mu_sha,
                "shift_acc_std": sd_sha,
                "shift_acc_ci95": ci_sha,
                "shift_acc_meanpmci": fmt_mean_ci(mu_sha, ci_sha),

                "target_f1_mean": mu_t,
                "target_f1_std": sd_t,
                "target_f1_ci95": ci_t,
                "target_f1_meanpmci": fmt_mean_ci(mu_t, ci_t),

                "target_acc_mean": mu_ta,
                "target_acc_std": sd_ta,
                "target_acc_ci95": ci_ta,
                "target_acc_meanpmci": fmt_mean_ci(mu_ta, ci_ta),
            }
        )

    df_agg = pd.DataFrame(agg_rows).sort_values("variant").reset_index(drop=True)

    out_table_csv = Path(args.out_table_csv)
    out_table_csv.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_table_csv, index=False)

    print(f"Wrote per-run CSV: {out_csv}")
    print(f"Wrote aggregated CSV: {out_table_csv}")


if __name__ == "__main__":
    main()
