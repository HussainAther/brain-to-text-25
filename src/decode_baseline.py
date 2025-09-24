# src/decode_baseline.py
import argparse, json, os, torch
from torch.utils.data import DataLoader
from dataset import BT25H5

def collapse_ctc(ids, blank):
    out, prev = [], None
    for i in ids:
        if i == blank:
            prev = None
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out

def make_batch(dl):
    for x, _ in dl:  # dataset returns (X, text)
        # pad time to its own length (batch=1 -> no pad needed)
        yield x

def main(a):
    # Load dataset (val or test file, or a root dir with --split)
    split = a.split
    ds = BT25H5(a.val, split=split, bin_ms=a.bin_ms, max_ms=a.max_ms)
    # Load checkpoint + vocab
    ck = torch.load(a.ckpt, map_location="cpu")
    vocab = ck["vocab"]; in_ch = ck.get("in_ch", ds[0][0].shape[0])
    blank = vocab.index("<BLANK>")
    # Build model (same as train)
    from train_baseline import EncoderCTC
    model = EncoderCTC(in_ch=in_ch, hidden=a.hidden, vocab=vocab)
    model.load_state_dict(ck["state"])
    model.eval()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    lines = ["id,text"]
    with torch.no_grad():
        for i,(X,_) in enumerate(dl):
            logp = model(X)                     # [1, T, V]
            pred = logp.argmax(-1).numpy()[0]   # [T]
            seq  = collapse_ctc(pred, blank)
            txt  = "".join(vocab[j] for j in seq)
            # Kaggle guidance: no punctuation; lowercase is OK
            lines.append(f"{i},{txt}")
            if a.max_items and i+1 >= a.max_items: break

    with open(a.out, "w") as f: f.write("\n".join(lines))
    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True, help="Path to data_val.hdf5 (or root dir).")
    p.add_argument("--split", default="val", choices=["val","test","train"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="outputs/val_predictions.csv")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=

