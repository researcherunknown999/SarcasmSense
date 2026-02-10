# SarcasmSense — Reproducible Experiments (Utterance-level MHCA + GMF)

This repository reproduces the manuscript experiments for **utterance-level** multimodal sarcasm detection using:
- Text encoder: **DeBERTa**
- Audio encoder: **Wav2Vec2 → CNN → BiLSTM**
- Video encoder: **TimeSformer** (+ optional **OpenFace FAU** token)
- Cross-modal alignment: **MHCA** (within the same utterance clip `u`)
- Fusion: **GMF** gating to produce `β(u)` and `E_final(u)`
- Multitask heads: **sarcasm** (binary), **sentiment shift** (binary), **rhetorical target** (K-way)

**Scope constraint:** All alignment and fusion are computed **per utterance**; the framework does **not** aggregate MHCA/GMF across multiple utterances or longer video segments.

---

## 1) Setup

### Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## One-command reproduction

Linux/macOS (Makefile):
```bash
make all DATA_ROOT=/path/to/DATA_ROOT DEVICE=cuda
```

Linux/macOS (bash):
```bash
bash run_all.sh --data_root /path/to/DATA_ROOT --seeds 1 2 3 4 5 6 7 8 9 10
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1 -DataRoot "C:\path\to\DATA_ROOT" -Seeds 1,2,3,4,5,6,7,8,9,10
```

Train a single variant/seed:
```bash
make train_one DATA_ROOT=/path/to/DATA_ROOT VARIANT=full_mhca_gmf_tav SEED=1
```

---

## 2) Data format

Place a `manifest.csv` inside your `DATA_ROOT` directory.

### Required columns
- `utt_id`
- `transcript_path` (relative to `DATA_ROOT`)
- `audio_path`
- `video_path`
- `y_sarc` (0/1)
- `y_shift` (0/1)
- `y_target` (0..K-1)

### Optional columns
- `split` in `{train,val,test}` (if missing, code creates stratified splits)
- `openface_path` (CSV with AU columns)

### Example manifest row
```csv
utt_id,split,transcript_path,audio_path,video_path,openface_path,y_sarc,y_shift,y_target
utt_0001,train,text/utt_0001.txt,audio/utt_0001.wav,video/utt_0001.mp4,openface/utt_0001.csv,1,1,2
```

---

## 3) Dataset + loader behavior

- **Utterance-level clips** `u` (1.5–30s): all modalities for the same `utt_id` are synchronized and processed **within** the clip.
- Text is tokenized with a DeBERTa tokenizer using `data.max_text_len`.
- Audio is loaded from WAV, resampled to `data.audio_sr`, padded/truncated to `data.audio_max_sec`.
- Video is uniformly sampled to `data.video_num_frames` frames and resized to `data.video_size`.
- OpenFace AU features (optional) are mean-pooled and appended as an **extra token** if enabled.

If the manifest has **no split**, the code creates **stratified** train/val/test splits by the joint label key:
`(y_sarc, y_shift, y_target)`.

---

## 4) Run one experiment (one seed)

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_root /path/to/DATA_ROOT \
  --run_dir artifacts/runs/full_mhca_gmf_tav/seed_1 \
  --seed 1 \
  --device cuda
```

### Overrides (ablation example)
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_root /path/to/DATA_ROOT \
  --run_dir artifacts/runs/wo_audio/seed_1 \
  --seed 1 \
  --override model.use_audio=false
```

---

## 5) Experiments reproduced

### Table 4: Main variants (unimodal / bimodal / multimodal)
```bash
python scripts/run_experiments.py \
  --data_root /path/to/DATA_ROOT \
  --device cuda \
  --exp_config configs/experiments_table4.yaml \
  --out_root artifacts/runs
```

### Table 7: Ablations (MHCA/GMF + modality removal)
```bash
python scripts/run_ablation.py \
  --data_root /path/to/DATA_ROOT \
  --device cuda \
  --exp_config configs/experiments_table7.yaml \
  --out_root artifacts/runs_ablation
```

### Table 3: Hyperparameter sensitivity (one-at-a-time)
```bash
python scripts/run_sensitivity.py \
  --data_root /path/to/DATA_ROOT \
  --device cuda \
  --exp_config configs/experiments_table3.yaml \
  --out_root artifacts/runs_sensitivity
```

---

## 6) Training details (reproducibility)

- Deterministic seeding via `sarcasmsense.utils.repro.set_global_seed`.
- Differential learning rates:
  - `train.lr_text` for the DeBERTa text encoder
  - `train.lr_av` for audio/video encoders + fusion + heads
- Scheduler: cosine with warmup (`train.warmup_ratio`)
- Early stopping on validation **mean macro-F1** across tasks.
- Threshold tuning:
  - Sarcasm and Shift thresholds are tuned on **VAL** using grid sweep to maximize macro-F1.
- Outputs per run directory:
  - `config_effective.json`
  - `summary.json` (thresholds, metrics, best epoch, best checkpoint path)
  - `train_history.json`
  - `checkpoints/best.pt`

---

## 7) Aggregate results (mean ± 95% CI across seeds)

Aggregate any runs root:
```bash
python scripts/aggregate_results.py --runs_root artifacts/runs \
  --out_csv artifacts/tables/table4_per_run.csv \
  --out_table_csv artifacts/tables/table4_agg.csv
```

Ablations:
```bash
python scripts/aggregate_results.py --runs_root artifacts/runs_ablation \
  --out_csv artifacts/tables/table7_per_run.csv \
  --out_table_csv artifacts/tables/table7_agg.csv
```

Sensitivity:
```bash
python scripts/aggregate_results.py --runs_root artifacts/runs_sensitivity \
  --out_csv artifacts/tables/table3_per_run.csv \
  --out_table_csv artifacts/tables/table3_agg.csv
```

---

## 8) Plots (confusion matrices)

Pick the best seed per variant (by `best_val_score`) and plot confusion matrices:
```bash
python scripts/plot_figures.py --runs_root artifacts/runs --split test --out_dir artifacts/figs/main
python scripts/plot_figures.py --runs_root artifacts/runs_ablation --split test --out_dir artifacts/figs/ablation
```

---

## 9) Environment check

Validate your environment and manifest quickly:
```bash
python scripts/check_env.py --data_root /path/to/DATA_ROOT
```

---

## 10) Notes on dataset privacy

This codebase is designed to work with **local** datasets via `manifest.csv`.
It does **not** require public release of the dataset to reproduce the experiments internally.

---

## License

MIT (see `LICENSE`).
