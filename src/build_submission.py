# src/build_submission.py
import argparse, os, torch, csv
from torch.utils.data import DataLoader
from dataset import BT25H5
from train_baseline import EncoderCTC
from utils import ctc_collapse

def greedy_decode(model, X, blank):
    with torch.no_grad():
        logp = model(X)                  # [1,T,V]
        pred = logp.argmax(-1)[0].cpu().numpy().tolist()
        return pred

def main(a):
    ds = BT25H5(a.test_h5, split="test", bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    ck = torch.load(a.ckpt, map_location="cpu")
    vocab = ck["vocab"]; in_ch = ck["in_ch"]; blank = vocab.index("<BLANK>")
    model = EncoderCTC(in_ch=in_ch, hidden=a.hidden, vocab=vocab)
    model.load_state_dict(ck["state"]); model.eval()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","text"])
        for i,(X,_) in enumerate(dl):
            ids = greedy_decode(model, X, blank)
            seq = ctc_collapse(ids, blank)
            text = "".join(vocab[j] for j in seq)
            w.writerow([i, text])
            if a.max_items and (i+1) >= a.max_items: break

    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_h5", required=True, help=".../t15.YYYY.MM.DD/data_test.hdf5")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--max_items", type=int, default=0)
    main(p.parse_args())
# src/build_submission.py
import argparse, os, torch, csv
from torch.utils.data import DataLoader
from dataset import BT25H5
from train_baseline import EncoderCTC
from utils import ctc_collapse

def greedy_decode(model, X, blank):
    with torch.no_grad():
        logp = model(X)                  # [1,T,V]
        pred = logp.argmax(-1)[0].cpu().numpy().tolist()
        return pred

def main(a):
    ds = BT25H5(a.test_h5, split="test", bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    ck = torch.load(a.ckpt, map_location="cpu")
    vocab = ck["vocab"]; in_ch = ck["in_ch"]; blank = vocab.index("<BLANK>")
    model = EncoderCTC(in_ch=in_ch, hidden=a.hidden, vocab=vocab)
    model.load_state_dict(ck["state"]); model.eval()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","text"])
        for i,(X,_) in enumerate(dl):
            ids = greedy_decode(model, X, blank)
            seq = ctc_collapse(ids, blank)
            text = "".join(vocab[j] for j in seq)
            w.writerow([i, text])
            if a.max_items and (i+1) >= a.max_items: break

    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_h5", required=True, help=".../t15.YYYY.MM.DD/data_test.hdf5")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--max_items", type=int, default=0)
    main(p.parse_args())

