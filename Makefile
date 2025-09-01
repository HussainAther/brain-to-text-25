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

preview:
\tBT25_DATA_DIR=$(DATA_DIR) BT25_OUT_DIR=$(OUT_DIR) $(ACT) && $(PY) main.py

clean:
\trm -rf $(OUT_DIR)
