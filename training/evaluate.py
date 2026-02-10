from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    BinaryThresholds,
    tune_binary_threshold,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_confusions,
    bootstrap_ci,
)


@torch.no_grad()
def _collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()

    ys_sarc, ys_shift, ys_target = [], [], []
    ps_sarc, ps_shift = [], []
    logits_target = []

    for batch in tqdm(loader, desc="eval", leave=False):
        # Move to device
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        audio_wav = batch.audio_wav.to(device)
        audio_mask = batch.audio_mask.to(device)
        video_frames = batch.video_frames.to(device)
        video_mask = batch.video_mask.to(device)
        openface_au = batch.openface_au.to(device)
        openface_cov = batch.openface_cov.to(device)

        y_sarc = batch.y_sarc.cpu().numpy()
        y_shift = batch.y_shift.cpu().numpy()
        y_target = batch.y_target.cpu().numpy()

        with torch.cuda.amp.autocast(enabled=bool(amp)):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_wav=audio_wav,
                audio_mask=audio_mask,
                video_frames=video_frames,
                video_mask=video_mask,
                openface_au=openface_au,
                openface_cov=openface_cov,
                return_aux=False,
            )

        # probs for positive class
        p_sarc = torch.softmax(out["logits_sarc"], dim=-1)[:, 1].detach().cpu().numpy()
        p_shift = torch.softmax(out["logits_shift"], dim=-1)[:, 1].detach().cpu().numpy()

        ys_sarc.append(y_sarc)
        ys_shift.append(y_shift)
        ys_target.append(y_target)
        ps_sarc.append(p_sarc)
        ps_shift.append(p_shift)
        logits_target.append(out["logits_target"].detach().cpu().numpy())

    y_sarc = np.concatenate(ys_sarc, axis=0)
    y_shift = np.concatenate(ys_shift, axis=0)
    y_target = np.concatenate(ys_target, axis=0)
    p_sarc = np.concatenate(ps_sarc, axis=0)
    p_shift = np.concatenate(ps_shift, axis=0)
    log_t = np.concatenate(logits_target, axis=0)

    return {
        "y_sarc": y_sarc,
        "y_shift": y_shift,
        "y_target": y_target,
        "p_sarc": p_sarc,
        "p_shift": p_shift,
        "logits_target": log_t,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Optional[BinaryThresholds] = None,
    tune_thresholds: bool = False,
    num_target_classes: Optional[int] = None,
    with_auc: bool = True,
    with_bootstrap_ci: bool = True,
    n_boot: int = 2000,
    seed: int = 0,
    amp: bool = False,
) -> Tuple[Dict[str, Any], BinaryThresholds]:
    """
    Evaluate model on loader.
    - If tune_thresholds=True, compute best thresholds on this split.
    Returns (metrics_dict, thresholds_used).
    """
    if thresholds is None:
        thresholds = BinaryThresholds()

    data = _collect_outputs(model, loader, device=device, amp=amp)

    y_sarc = data["y_sarc"]
    y_shift = data["y_shift"]
    y_target = data["y_target"]
    p_sarc = data["p_sarc"]
    p_shift = data["p_shift"]
    log_t = data["logits_target"]
    y_target_pred = np.argmax(log_t, axis=-1)

    # Tune thresholds on this split (typically VAL)
    if tune_thresholds:
        thr_sarc, _ = tune_binary_threshold(y_sarc, p_sarc)
        thr_shift, _ = tune_binary_threshold(y_shift, p_shift)
        thresholds = BinaryThresholds(sarc=thr_sarc, shift=thr_shift)

    # Apply thresholds
    y_sarc_pred = (p_sarc >= thresholds.sarc).astype(int)
    y_shift_pred = (p_shift >= thresholds.shift).astype(int)

    m_sarc = compute_binary_metrics(y_sarc, p_sarc, thresholds.sarc, with_auc=with_auc)
    m_shift = compute_binary_metrics(y_shift, p_shift, thresholds.shift, with_auc=with_auc)
    m_target = compute_multiclass_metrics(y_target, y_target_pred)

    # CIs (bootstrap on macro-F1 using *final* discrete predictions)
    ci = {}
    if with_bootstrap_ci:
        ci["sarc_macro_f1_ci95"] = bootstrap_ci(
            y_sarc, y_sarc_pred,
            metric_fn=lambda yt, yp: float(np.mean([1.0])) if len(yt) == 0 else float(
                __import__("sklearn.metrics").metrics.f1_score(yt, yp, average="macro")
            ),
            n_boot=n_boot,
            seed=seed + 11,
        )
        ci["shift_macro_f1_ci95"] = bootstrap_ci(
            y_shift, y_shift_pred,
            metric_fn=lambda yt, yp: float(__import__("sklearn.metrics").metrics.f1_score(yt, yp, average="macro")),
            n_boot=n_boot,
            seed=seed + 22,
        )
        ci["target_macro_f1_ci95"] = bootstrap_ci(
            y_target, y_target_pred,
            metric_fn=lambda yt, yp: float(__import__("sklearn.metrics").metrics.f1_score(yt, yp, average="macro")),
            n_boot=n_boot,
            seed=seed + 33,
        )

    cms = compute_confusions(
        y_sarc, y_sarc_pred,
        y_shift, y_shift_pred,
        y_target, y_target_pred,
        num_target_classes=num_target_classes,
    )

    # Aggregate score for early stopping (mean macro-F1 across tasks)
    mean_macro_f1 = float(np.nanmean([m_sarc["macro_f1"], m_shift["macro_f1"], m_target["macro_f1"]]))

    metrics: Dict[str, Any] = {
        "thresholds": asdict(thresholds),
        "sarc": m_sarc,
        "shift": m_shift,
        "target": m_target,
        "mean_macro_f1": mean_macro_f1,
        "confusions": {k: v.tolist() for k, v in cms.items()},
        "ci95": {k: [float(ci[k][0]), float(ci[k][1])] for k in ci} if with_bootstrap_ci else {},
        "n": int(len(y_sarc)),
    }
    return metrics, thresholds
