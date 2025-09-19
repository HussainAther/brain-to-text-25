# Brain-to-Text ’25 – Minimal Baseline

End-to-end, reproducible starter for the Kaggle **Brain-to-Text ’25** challenge:

* Load **T15** intracortical datasets from the zipped release (multi-session tree of HDF5s).
* Train a tiny **CTC** baseline that accepts variable **channels** and **time** lengths.
* Decode to CSV (for quick sanity checks).

> This repo is intentionally minimal and readable. It’s a foundation we’ll iterate on (better features, models, and WER eval) as we onboard contributors.

---

## Repo layout

```
.
├─ src/
│  ├─ dataset.py            # Robust loader for T15 HDF5s (handles per-trial groups; infers [C,T])
│  ├─ train_baseline.py     # CTC baseline, pads both channels & time; --max_batches for smoke tests
│  └─ decode_baseline.py    # Greedy decode -> CSV (uses saved vocab)
├─ data/
│  └─ brain-to-text-25.zip  # (downloaded via Kaggle CLI) → unzip here
├─ Makefile                 # Convenience targets (optional)
├─ config.yaml              # (optional) placeholder for future configs
└─ README.md
```

---

## 1) Setup

### Conda (recommended)

```bash
conda create -n bt25 python=3.10 -y
conda activate bt25
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or CUDA wheel on MedARC
pip install h5py numpy matplotlib tqdm
# (Optional) kaggle CLI if you’ll download from Kaggle directly
pip install kaggle
```

> On MedARC, use the CUDA wheel (or the preinstalled module) so `torch.cuda.is_available()` is `True`.

---

## 2) Get the data

If you already have the zip, put it at `./data/brain-to-text-25.zip`, then:

```bash
cd data
unzip brain-to-text-25.zip -d brain-to-text
cd ..
```

The important subtree will look like:

```
data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final/t15.YYYY.MM.DD/data_{train,val,test}.hdf5
```

Export a root env var (adjust path if yours differs):

```bash
export BT25_ROOT="$PWD/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final"
```

If you prefer Kaggle CLI:

```bash
mkdir -p data
kaggle competitions download -c brain-to-text-25 -p data
unzip data/brain-to-text-25.zip -d data/brain-to-text
export BT25_ROOT="$PWD/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final"
```

---

## 3) Quick smoke test (CPU or GPU)

### Single session (quickest)

```bash
python src/train_baseline.py \
  --train "$BT25_ROOT/t15.2023.10.06/data_train.hdf5" \
  --epochs 1 \
  --max_batches 2
```

What you should see:

* Device printout (CPU or GPU).
* Progress logs every \~25 steps (here you’ll exit early at 2 batches).
* A saved checkpoint: `checkpoints/baseline.pt` and `metadata_vocab.json`.

### Multi-session (auto-scan all `t15.*` folders)

```bash
python src/train_baseline.py \
  --data_dir "$BT25_ROOT" \
  --epochs 1 \
  --batch 4 \
  --max_batches 5
```

> The training script **pads both channels and time** and infers a safe conv input size.
> If you ever hit a “channel size” mismatch across sessions, raise the cap:
> `--channel_cap 1536` (default is `1024`).

---

## 4) Decode a validation file (sanity check)

```bash
python src/decode_baseline.py \
  --val "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" \
  --ckpt checkpoints/baseline.pt \
  --out outputs/val_predictions.csv
```

This writes a CSV like:

```
id,text
0,hello world
1, ...
```

(We’ll wire formal WER evaluation next.)

---

## Notes on the loader (`src/dataset.py`)

* Handles **per-trial HDF5 layouts** (e.g., `trial_0000/...`) and common alternatives:

  * `neural_data`, `binned_spikes`, `features`, `rates` datasets
  * `spike_times` groups (per-channel or packed arrays)
* Auto-orients trials to **\[C, T]** (channels × time), regardless of original shape.
* Bins spike times at **10 ms** (configurable: `--bin_ms` / `--max_ms` in `train_baseline.py`).

---

## MedARC compute (GPU quick-check)

SSH / Jupyter into MedARC; then run:

```bash
# verify GPU
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# 1-batch smoke test
export BT25_ROOT="/path/to/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final"
python src/train_baseline.py --train "$BT25_ROOT/t15.2023.10.06/data_train.hdf5" --epochs 1 --max_batches 1
```

---

## Roadmap

* ✅ Robust multi-session loader, variable-length padding, smoke tests
* ⏳ WER evaluation script (align decoded CSV with ground-truth)
* ⏳ Phoneme posterior baseline (to match Kaggle starter more closely)
* ⏳ LM rescoring (n-gram → transformer)
* ⏳ Config-driven experiments + logging

---

## Contributors

* **Syed Hussain Ather** – Lead
* **Anirudh Gangadharan** – Research & Ops

We welcome MedARC collaborators after soft launch (Sept 24).

---

## Troubleshooting

* **File not found**: double-check `BT25_ROOT` and unzip destination.
* **Different channel counts** across sessions: increase `--channel_cap` (e.g., 1536).
* **Slow on laptop**: use `--max_batches 2` to smoke test locally; do full runs on MedARC GPUs.

---

### One-liner Quickstart (after unzip)

```bash
export BT25_ROOT="$PWD/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final" && \
python src/train_baseline.py --train "$BT25_ROOT/t15.2023.10.06/data_train.hdf5" --epochs 1 --max_batches 2 && \
python src/decode_baseline.py --val "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" --ckpt checkpoints/baseline.pt --out outputs/val_predictions.csv
