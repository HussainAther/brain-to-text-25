# 🧠 Brain-to-Text ’25 — Minimal Working Baseline (MedARC)

This repository contains the current **baseline pipeline** for converting intracortical neural recordings into text using open datasets from the **Brain-to-Text 2025 Kaggle Challenge**.  
The code supports end-to-end training, decoding, rescoring, and evaluation on the official `t15_copyTask_neuralData` recordings.

---

## 📁 Project Structure
```

brain-to-text/
├── src/
│   ├── train_baseline.py       # BiLSTM-CTC model training
│   ├── decode_baseline.py      # Greedy decoding
│   ├── decode_rescore.py       # Beam search + LM rescoring
│   ├── eval_wer.py             # WER evaluation (Kaggle-style)
│   ├── lm_char_train.py        # Character-level LM (GRU)
│   ├── channel_select.py       # Feature selection for top-K electrodes
│   ├── dataset.py              # HDF5 loader + spike binning
│   └── inspect_h5.py           # Quick trial inspection / debugging
├── config/
│   └── baseline.yaml           # Experiment hyperparameters
├── checkpoints/                # Saved models (.pt)
├── outputs/                    # Decoded CSVs and logs
├── artifacts/                  # Channel masks etc.
└── README.md

````

---

## ⚙️ Environment Setup
```bash
conda create -n bt25 python=3.10 -y
conda activate bt25
pip install torch h5py numpy tqdm pandas
````

If running from an **external drive**:

```bash
export BT25_ROOT="/Volumes/External/brain-to-text/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final"
```

---

## 🚀 Quick Start

Train → Decode → Evaluate:

```bash
# Train baseline BiLSTM-CTC
python src/train_baseline.py \
  --train "$BT25_ROOT/t15.2023.10.06/data_train.hdf5" \
  --epochs 1 --max_batches 100

# Greedy decoding
python src/decode_baseline.py \
  --val "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" \
  --ckpt checkpoints/baseline.pt \
  --out outputs/val_greedy.csv

# Evaluate
python src/eval_wer.py \
  --ref_h5 "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" \
  --pred_csv outputs/val_greedy.csv
```

---

## 🧩 Optional Improvements

### 🔤 1. Train a small character-level LM

```bash
python src/lm_char_train.py --data_root "$BT25_ROOT" --epochs 1
```

### 🪶 2. LM-rescored decoding

```bash
python src/decode_rescore.py \
  --val "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" \
  --ckpt checkpoints/baseline.pt \
  --lm_ckpt checkpoints/lm_char.pt \
  --beam 12 --lm_weight 0.75 \
  --out outputs/val_lm.csv
```

### ⚗️ 3. Evaluate WER after rescoring

```bash
python src/eval_wer.py \
  --ref_h5 "$BT25_ROOT/t15.2023.10.06/data_val.hdf5" \
  --pred_csv outputs/val_lm.csv
```

### 🧮 4. Electrode channel selection

```bash
python src/channel_select.py \
  --h5 "$BT25_ROOT/t15.2023.10.06/data_train.hdf5" \
  --keep 256
```

`dataset.py` automatically loads `artifacts/channels_keep.npy` if it exists.

---

## 📊 Baseline Model

| Component | Description                                 |
| --------- | ------------------------------------------- |
| Encoder   | 2-layer **BiLSTM** with 256 hidden units    |
| Output    | **CTC loss** over character vocabulary      |
| Input     | Binned spike trains (10 ms bins, up to 3 s) |
| Decoder   | Greedy or Beam + Char-LM rescoring          |
| Metric    | Word Error Rate (WER)                       |

---

## 🧠 Roadmap

* [x] End-to-end baseline pipeline
* [x] Character-LM rescoring
* [x] Channel feature selection
* [ ] Fine-tuning + model compression
* [ ] Integrate EEG/ECoG-to-speech demo for Quantum Nets AI relevance

---

## 🧩 Maintainers

* **Syed Ather** – Lead Engineer
* **Paul [MedARC]** – Project Coordinator
* Collaborators: MedARC Neuro AI Team

---

## 📜 Citation

If you use this baseline, please cite the Brain-to-Text 2025 Kaggle dataset and MedARC Lab’s work on neural decoding.

