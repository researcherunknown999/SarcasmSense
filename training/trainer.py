from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .loss import LossWeights, MultiTaskLoss
from .evaluate import evaluate
from .metrics import BinaryThresholds
from ..utils.io import ensure_dir, save_json


class Trainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        device: torch.device,
        run_dir: str | Path,
        seed: int,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.seed = int(seed)

        self.tcfg = cfg["train"]
        self.mcfg = cfg["model"]

        ensure_dir(self.run_dir)
        ensure_dir(self.run_dir / "checkpoints")

        self.model.to(self.device)

        lw = LossWeights(
            sarc=float(self.tcfg.get("alpha_sarc", 0.60)),
            shift=float(self.tcfg.get("alpha_shift", 0.25)),
            target=float(self.tcfg.get("alpha_target", 0.15)),
        )
        self.criterion = MultiTaskLoss(lw)

        self.amp = bool(self.tcfg.get("amp", True)) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.grad_clip = float(self.tcfg.get("grad_clip", 1.0))
        self.patience = int(self.tcfg.get("early_stop_patience", 5))
        self.max_epochs = int(self.tcfg.get("max_epochs", 30))

        # thresholds tuned on val
        self.thresholds = BinaryThresholds()

        # Optimizer & scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = None  # built in fit after steps known

        # bookkeeping
        self.best_val = -1e9
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_ckpt = self.run_dir / "checkpoints" / "best.pt"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr_text = float(self.tcfg.get("lr_text", 3e-5))
        lr_av = float(self.tcfg.get("lr_av", 1e-4))
        wd = float(self.tcfg.get("weight_decay", 0.01))

        # Parameter groups: text vs (audio+video+fusion+heads)
        text_params = list(self.model.text_enc.parameters()) if hasattr(self.model, "text_enc") else []
        text_param_ids = {id(p) for p in text_params}

        other_params = [p for p in self.model.parameters() if id(p) not in text_param_ids]

        groups = [
            {"params": text_params, "lr": lr_text, "weight_decay": wd},
            {"params": other_params, "lr": lr_av, "weight_decay": wd},
        ]
        return AdamW(groups)

    def _build_scheduler(self, num_training_steps: int) -> None:
        warmup_ratio = float(self.tcfg.get("warmup_ratio", 0.10))
        num_warmup = int(math.floor(warmup_ratio * num_training_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        t0 = time.time()

        total = 0.0
        ls = 0.0
        lsh = 0.0
        lt = 0.0
        n = 0

        for batch in tqdm(loader, desc=f"train e{epoch}", leave=False):
            input_ids = batch.input_ids.to(self.device)
            attention_mask = batch.attention_mask.to(self.device)
            audio_wav = batch.audio_wav.to(self.device)
            audio_mask = batch.audio_mask.to(self.device)
            video_frames = batch.video_frames.to(self.device)
            video_mask = batch.video_mask.to(self.device)
            openface_au = batch.openface_au.to(self.device)
            openface_cov = batch.openface_cov.to(self.device)

            y_sarc = batch.y_sarc.to(self.device)
            y_shift = batch.y_shift.to(self.device)
            y_target = batch.y_target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(
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
                loss_dict = self.criterion(out, y_sarc=y_sarc, y_shift=y_shift, y_target=y_target)
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            bs = int(input_ids.shape[0])
            total += float(loss.detach().cpu().item()) * bs
            ls += float(loss_dict["sarc"].detach().cpu().item()) * bs
            lsh += float(loss_dict["shift"].detach().cpu().item()) * bs
            lt += float(loss_dict["target"].detach().cpu().item()) * bs
            n += bs

        dt = time.time() - t0
        return {
            "loss_total": total / max(1, n),
            "loss_sarc": ls / max(1, n),
            "loss_shift": lsh / max(1, n),
            "loss_target": lt / max(1, n),
            "sec": dt,
        }

    def fit(self, loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Train with early stopping on VAL mean_macro_f1 (mean macro-F1 across tasks).
        Saves best checkpoint and summary.json with:
          - thresholds tuned on VAL
          - VAL and TEST metrics
        """
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]

        # scheduler needs total steps
        steps_per_epoch = max(1, len(train_loader))
        num_steps = steps_per_epoch * self.max_epochs
        self._build_scheduler(num_training_steps=num_steps)

        history = []
        for epoch in range(1, self.max_epochs + 1):
            tr = self._train_one_epoch(train_loader, epoch)

            # Tune thresholds on VAL, then evaluate VAL with tuned thresholds
            val_metrics, tuned_thr = evaluate(
                self.model,
                val_loader,
                device=self.device,
                thresholds=None,
                tune_thresholds=True,
                num_target_classes=int(self.mcfg.get("num_target_classes", 5)),
                with_auc=bool(self.tcfg.get("with_auc", True)),
                with_bootstrap_ci=bool(self.tcfg.get("with_bootstrap_ci", True)),
                n_boot=int(self.tcfg.get("n_boot", 2000)),
                seed=self.seed,
                amp=self.amp,
            )
            self.thresholds = tuned_thr

            score = float(val_metrics["mean_macro_f1"])
            rec = {"epoch": epoch, "train": tr, "val": val_metrics, "val_score": score}
            history.append(rec)

            # Early stopping
            if score > self.best_val:
                self.best_val = score
                self.best_epoch = epoch
                self.bad_epochs = 0
                torch.save(
                    {"model": self.model.state_dict(), "cfg": self.cfg, "epoch": epoch, "thresholds": asdict(self.thresholds)},
                    self.best_ckpt,
                )
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                break

        # Load best and evaluate on VAL + TEST with best thresholds
        ck = torch.load(self.best_ckpt, map_location=self.device)
        self.model.load_state_dict(ck["model"], strict=True)
        self.thresholds = BinaryThresholds(**ck.get("thresholds", {"sarc": 0.5, "shift": 0.5}))

        val_final, _ = evaluate(
            self.model, val_loader, device=self.device,
            thresholds=self.thresholds, tune_thresholds=False,
            num_target_classes=int(self.mcfg.get("num_target_classes", 5)),
            with_auc=bool(self.tcfg.get("with_auc", True)),
            with_bootstrap_ci=bool(self.tcfg.get("with_bootstrap_ci", True)),
            n_boot=int(self.tcfg.get("n_boot", 2000)),
            seed=self.seed + 101,
            amp=self.amp,
        )
        test_final, _ = evaluate(
            self.model, test_loader, device=self.device,
            thresholds=self.thresholds, tune_thresholds=False,
            num_target_classes=int(self.mcfg.get("num_target_classes", 5)),
            with_auc=bool(self.tcfg.get("with_auc", True)),
            with_bootstrap_ci=bool(self.tcfg.get("with_bootstrap_ci", True)),
            n_boot=int(self.tcfg.get("n_boot", 2000)),
            seed=self.seed + 202,
            amp=self.amp,
        )

        summary = {
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_val_score": float(self.best_val),
            "thresholds": asdict(self.thresholds),
            "val": val_final,
            "test": test_final,
            "cfg": self.cfg,  # full config for reproducibility
            "best_ckpt": str(self.best_ckpt),
        }
        save_json(self.run_dir / "summary.json", summary)
        save_json(self.run_dir / "train_history.json", {"history": history})

        return summary
