#!/usr/bin/env python
"""
Quick & dirty Brain-to-Text '25 Kaggle submission script.

- Uses the official pretrained GRUDecoder baseline
- Runs on the test split of all sessions
- Greedy-decodes phoneme sequences (no n-gram LM, no OPT)
- Writes a Kaggle-style submission.csv with columns: id,text

NOTE: The "text" here is actually a sequence of ARPABET phonemes,
      not proper English sentences. WER will be bad, but it's a
      valid submission and closes the loop.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Import baseline model + helpers from the repo
from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import (
    load_h5py_file,
    LOGIT_TO_PHONEME,
    runSingleDecodingStep,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a phoneme-only Kaggle submission using the pretrained baseline RNN."
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="data/t15_pretrained_rnn_baseline",
        help="Path to pretrained model directory (with checkpoint/args.yaml).",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data/hdf5_data_final",
        help="Path to hdf5_data_final root containing t15.* session dirs.",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default="data/t15_copyTaskData_description.csv",
        help="Path to CSV metadata file.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV path (Kaggle submission).",
    )
    p.add_argument(
        "--eval_type",
        type=str,
        default="test",
        choices=["val", "test"],
        help='Which split to decode. For Kaggle, use "test".',
    )
    p.add_argument(
        "--gpu_number",
        type=int,
        default=-1,
        help="GPU index to use. Set -1 for CPU.",
    )
    return p


def load_model(model_path: str, device: torch.device):
    """Load GRUDecoder and its args from checkpoint/args.yaml + best_checkpoint."""
    args_path = os.path.join(model_path, "checkpoint", "args.yaml")
    ckpt_path = os.path.join(model_path, "checkpoint", "best_checkpoint")

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Missing args.yaml at {args_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing best_checkpoint at {ckpt_path}")

    model_args = OmegaConf.load(args_path)

    model = GRUDecoder(
        neural_dim=model_args["model"]["n_input_features"],
        n_units=model_args["model"]["n_units"],
        n_days=len(model_args["dataset"]["sessions"]),
        n_classes=model_args["dataset"]["n_classes"],
        rnn_dropout=model_args["model"]["rnn_dropout"],
        input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
        n_layers=model_args["model"]["n_layers"],
        patch_size=model_args["model"]["patch_size"],
        patch_stride=model_args["model"]["patch_stride"],
    )

    # Load checkpoint and strip DataParallel prefixes if present
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"]
    cleaned = {}
    for k, v in state.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_k] = v
    model.load_state_dict(cleaned)

    model.to(device)
    model.eval()

    return model, model_args


def choose_device(gpu_number: int) -> torch.device:
    if torch.cuda.is_available() and gpu_number >= 0:
        if gpu_number >= torch.cuda.device_count():
            raise ValueError(
                f"GPU number {gpu_number} is out of range "
                f"(available: {torch.cuda.device_count()})"
            )
        device = torch.device(f"cuda:{gpu_number}")
        print(f"[device] Using {device} for inference.")
    else:
        if gpu_number >= 0 and not torch.cuda.is_available():
            print(f"[device] GPU {gpu_number} requested but CUDA not available; using CPU.")
        device = torch.device("cpu")
        print(f"[device] Using CPU.")
    return device


def main():
    parser = build_argparser()
    args = parser.parse_args()

    device = choose_device(args.gpu_number)

    # Load dataset metadata CSV
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"[data] CSV metadata not found: {args.csv_path}")
    meta_df = pd.read_csv(args.csv_path)

    # Load model + its Hydra args
    model, model_args = load_model(args.model_path, device)

    # Collect test data from each session listed in model_args
    sessions = model_args["dataset"]["sessions"]
    eval_type = args.eval_type  # "test" for Kaggle submission

    all_predictions = []
    all_ids = []

    total_trials = 0
    test_data = {}

    # 1) Load data (neural features) for each session
    for session in sessions:
        sess_dir = os.path.join(args.data_dir, session)
        if not os.path.isdir(sess_dir):
            print(f"[warn] Session directory not found, skipping: {sess_dir}")
            continue

        h5_name = f"data_{eval_type}.hdf5"
        h5_path = os.path.join(sess_dir, h5_name)
        if not os.path.exists(h5_path):
            print(f"[warn] {h5_name} missing in {sess_dir}, skipping.")
            continue

        data = load_h5py_file(h5_path, meta_df)
        test_data[session] = data
        n_trials = len(data["neural_features"])
        total_trials += n_trials
        print(f"[data] Loaded {n_trials} {eval_type} trials for session {session}.")

    print(f"[data] Total {eval_type} trials across all sessions: {total_trials}")

    # 2) Forward pass through the RNN to get logits
    print("[model] Running inference to obtain phoneme logits...")

    with tqdm(total=total_trials, desc="Decoding", unit="trial") as pbar:
        for session, data in test_data.items():
            # Identify which "day index" this session corresponds to
            input_layer = sessions.index(session)
            n_trials = len(data["neural_features"])

            for trial_idx in range(n_trials):
                neural_input = data["neural_features"][trial_idx]  # shape [T, 512]
                # Add batch dimension
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(
                    neural_input, device=device, dtype=torch.bfloat16
                )

                # runSingleDecodingStep applies smoothing & model forward
                logits = runSingleDecodingStep(
                    neural_input, input_layer, model, model_args, device
                )  # numpy, shape [1, T', n_classes]

                logits = logits[0]  # [T', n_classes]

                # Greedy decode
                pred_ids = np.argmax(logits, axis=-1).tolist()  # list of class indices

                # Remove blanks (index 0 by convention)
                pred_ids = [p for p in pred_ids if p != 0]

                # Collapse repeats
                collapsed = []
                prev = None
                for p in pred_ids:
                    if p != prev:
                        collapsed.append(p)
                    prev = p

                # Map to phoneme strings
                phonemes = [LOGIT_TO_PHONEME[p] for p in collapsed if p < len(LOGIT_TO_PHONEME)]

                # Join as a space-separated "sentence"
                text = " ".join(phonemes).strip()

                # Append to global lists
                all_predictions.append(text)
                all_ids.append(len(all_ids))  # sequential id

                pbar.update(1)

    # 3) Write Kaggle-style CSV
    print(f"[out] Writing {len(all_predictions)} predictions to {args.output}")
    df_out = pd.DataFrame({"id": all_ids, "text": all_predictions})
    df_out.to_csv(args.output, index=False)
    print("[done] Submission file written.")


if __name__ == "__main__":
    main()

