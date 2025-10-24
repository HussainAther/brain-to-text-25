SHELL := /bin/bash
PY := python
VENV := .venv
ACT := source $(VENV)/bin/activate

DATA_DIR ?= data
OUT_DIR ?= outputs

.PHONY: help setup install download preview clean

help:
\t@echo "Make targets:"
\t@echo "  make setup      - create venv and install requirements"
\t@echo "  make install    - install requirements into existing venv"
\t@echo "  make download   - download Kaggle Brain-to-Text '25 to $(DATA_DIR)"
\t@echo "  make preview    - run main.py (loads HDF5 and saves a heatmap to $(OUT_DIR))"
\t@echo "  make clean      - remove outputs/"

setup:
\t$(PY) -m venv $(VENV)
\t$(ACT) && pip install --upgrade pip
\t$(ACT) && pip install -r requirements.txt

install:
\t$(ACT) && pip install -r requirements.txt

download:
\tBT25_DATA_DIR=$(DATA_DIR) $(ACT) && $(PY) main.py

multi:
\tpython src/train_multisession.py --root "$(BT25_ROOT)" --epochs 2 --max_batches 500

submit-one:
\tpython src/build_submission.py --test_h5 "$(BT25_ROOT)/t15.2023.10.06/data_test.hdf5" --ckpt checkpoints/baseline.pt --out outputs/submission_2023-10-06.csv

quicklook:
\tpython src/plot_quicklook.py "$(BT25_ROOT)/t15.2023.10.06/data_train.hdf5" --trial 0 --out outputs/quicklook.png

preview:
\tBT25_DATA_DIR=$(DATA_DIR) BT25_OUT_DIR=$(OUT_DIR) $(ACT) && $(PY) main.py

kenlm-corpus:
\tpython src/lm_kenlm_build.py --root "$(BT25_ROOT)" --corpus artifacts/corpus.txt

kenlm-build:
\tpython src/lm_kenlm_build.py --root "$(BT25_ROOT)" --build --order 5 \
\t  --corpus artifacts/corpus.txt --arpa artifacts/lm.arpa --binary artifacts/lm.binary

decode-kenlm:
\tpython src/decode_kenlm.py --val "$(BT25_ROOT)/t15.2023.10.06/data_val.hdf5" \
\t  --ckpt checkpoints/baseline.pt --beam 12 --lm_weight 0.8 \
\t  --out outputs/val_kenlm.csv


clean:
\trm -rf $(OUT_DIR)

