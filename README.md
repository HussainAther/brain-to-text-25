# 🧠 Brain-to-Text '25 – Neural Speech Decoding

This repository contains our codebase for the [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) Kaggle competition, which challenges participants to decode intended speech from intracortical neural activity recorded in the speech motor cortex.

## 🧑‍💻 Team
- [Syed Hussain Ather](https://www.linkedin.com/in/syed-hussain-ather-049919137/) | [GitHub](https://github.com/hussainather)
- [Anirudh Gangadharan](https://www.linkedin.com/in/anirudh-gangadharan-792a61286/)

## 🧭 Project Overview

> People with conditions like ALS can lose their ability to speak. This project aims to decode what they're trying to say directly from their brain signals using machine learning.

We are building and experimenting with deep learning models (RNNs, Transformers, CTC loss, language model rescoring, etc.) to improve speech decoding accuracy from brain recordings.

---

## 📁 Project Structure

```

brain-to-text-25/
├── data/                  # Local data folder (not committed)
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # Core scripts and model code
│   ├── data\_loader.py     # Spike data loading and binning
│   ├── model.py           # Model architecture(s)
│   ├── train.py           # Training loop
│   └── decode.py          # Decoding and WER evaluation
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore             # Ignore large files, data, and cache

````

---

## 📦 Setup

### Clone the repo
```bash
git clone https://github.com/HussainAther/brain-to-text-25.git
cd brain-to-text-25
````

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset is provided as `.hdf5` files by the competition organizers. Due to size and licensing, it is **not included in this repository**.

👉 Download the dataset from the official [Kaggle competition page](https://www.kaggle.com/competitions/brain-to-text-25/data), then place the files in the `data/` folder.

---

## 🚀 How to Run

### Run Data Inspection

```bash
python src/data_loader.py
```

### Train a Baseline Model

```bash
python src/train.py --config configs/baseline.yaml
```

### Evaluate WER

```bash
python src/decode.py --checkpoint model.pt
```

---

## ✅ TODO

* [x] Build data pipeline
* [ ] Build baseline model (LSTM + CTC)
* [ ] Add phoneme decoding
* [ ] Integrate language model rescoring
* [ ] Submit to leaderboard

---

## 📜 License & Citation

Please cite the NEJM 2024 paper if you use this work:

> Card et al., "An Accurate and Rapidly Calibrating Speech Neuroprosthesis", *New England Journal of Medicine*, 2024

---

## 🧠 Acknowledgments

Thanks to the UC Davis Neuroprosthetics Lab and BrainGate consortium for making this challenge and dataset available.


