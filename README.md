# Brain-to-Text ’25 Baseline (MedARC Project)

## 🔎 Project Overview

This project explores how to decode neural activity into text using the [Brain-to-Text ’25 Kaggle Challenge](https://www.kaggle.com/competitions/brain-to-text-25).

We are building a baseline pipeline that:

* Loads neural recordings from intracortical electrodes.
* Trains a recurrent model to map spikes → phonemes/words.
* Uses language model rescoring to reduce word error rate (WER).
* Evaluates predictions against ground truth transcripts.

This is a **MedARC community project** led by Syed Hussain Ather and Anirudh Gangadharan, with support from Paul Scotti and Tanishq Abraham.

---

## ⚙️ Setup

### Clone + environment

```bash
git clone https://github.com/YOUR_ORG/brain-to-text.git
cd brain-to-text
conda env create -f environment.yml
conda activate orel
```

### Kaggle credentials

You need a `kaggle.json` file in `~/.kaggle/`.
Download it from [Kaggle Account Settings](https://www.kaggle.com/account).

---

## 📂 Dataset

### Download

```bash
kaggle competitions download -c brain-to-text-25 -p data
unzip data/brain-to-text-25.zip -d data/
```

### Expected structure

```
brain-to-text/
  ├── src/
  ├── data/
  │   └── brain-to-text/
  │       └── t15_copyTask_neuralData/hdf5_data_final/...
```

### Environment variable

Point scripts to dataset root:

```bash
export BT25_ROOT="/path/to/brain-to-text/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final"
```

---

## 🚀 Running the Baseline

### Train model

```bash
python src/train_baseline.py \
  --train $BT25_ROOT/t15.2023.10.06/data_train.hdf5 \
  --epochs 10
```

### Decode with rescoring

```bash
python src/decode_rescore.py \
  --checkpoint checkpoints/baseline.pt \
  --test $BT25_ROOT/t15.2023.10.06/data_test.hdf5
```

### Evaluate WER

```bash
python src/eval_wer.py \
  --pred outputs/preds.txt \
  --ref outputs/refs.txt
```

---

## 🧪 Features in Progress

* Electrode feature selection (drop irrelevant channels).
* JEPA/TS-JEPA architectures for embeddings.
* Demo of hospital relevance (speech restoration for patients).

---

## 👥 Contributors

* **Syed Hussain Ather** — Lead
* **Anirudh Gangadharan** — Support & Ops
* **MedARC community** — Discussion, compute, clinical alignment

---

## 📅 Timeline

* **August 2025** — Baseline training & decoding scripts working.
* **September 2025** — Community onboarding, add language model rescoring.
* **Fall 2025** — Feature selection + JEPA embeddings.
* **Winter 2025** — Hospital demo + final Kaggle submission.

