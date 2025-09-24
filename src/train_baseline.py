# src/train_baseline.py
import argparse, json, os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BT25H5

class EncoderCTC(nn.Module):
    def __init__(self, in_ch=256, hidden=256, vocab=()):
        super().__init__()
        self.vocab = list(vocab)
        self.blank = self.vocab.index("<BLANK>")
        self.conv = nn.Conv1d(in_ch, hidden, 3, padding=1)
        self.rnn = nn.LSTM(hidden, hidden, 2, bidirectional=True, batch_first=True)
        self.fc  = nn.Linear(2*hidden, len(vocab))
    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x).permute(0,2,1)  # -> [B, T, H]
        x,_ = self.rnn(x)                # -> [B, T, 2H]
        return x.log_softmax(-1)         # -> [B, T, V]

def make_vocab(samples):
    symbols = sorted(set("".join(s.lower() for s in samples) + " "))
    vocab = ["<BLANK>"] + symbols
    return vocab, {c:i for i,c in enumerate(vocab)}

def encode_targets(texts, stoi):
    seqs = [[stoi.get(c, 0) for c in t.lower()] for t in texts]
    flat = [i for s in seqs for i in s]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.int32)
    return torch.tensor(flat, dtype=torch.int64), lengths

def collate(batch):
    # batch: list of (X[C,T], text)
    xs, ys = zip(*batch)
    C = xs[0].shape[0]
    Ts = [x.shape[1] for x in xs]
    maxT = max(Ts)
    Xpad = torch.zeros(len(xs), C, maxT, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        Xpad[i, :, :x.shape[1]] = x
    tlen = torch.tensor(Ts, dtype=torch.int32)  # per-sample input length
    return Xpad, list(ys), tlen

def main(args):
    # Accept either a file (--train) or a root dir (--data_dir)
    path = args.data_dir if args.data_dir else args.train
    if not path:
        raise SystemExit("Provide --data_dir (root) or --train (single data_train.hdf5).")

    ds = BT25H5(path, split="train", n_channels=args.channels, bin_ms=10, max_ms=3000)
    # Peek one sample to infer channels (handles 256 vs 512, etc.)
    X0, _ = ds[0]
    in_ch = X0.shape[0]

    # Build small vocab from first N examples
    n = min(256, len(ds))
    sample_texts = [ds[i][1] for i in range(n)]
    vocab, stoi = make_vocab(sample_texts)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    model = EncoderCTC(in_ch=in_ch, hidden=args.hidden, vocab=vocab).to(args.device)
    ctc = nn.CTCLoss(blank=model.blank, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for X, texts, tlen in dl:
            X = X.to(args.device)                           # [B,C,T]
            logp = model(X)                                 # [B,T,V]
            T_input = tlen.to(args.device)                  # true lengths per sample
            y, ylen = encode_targets(texts, {c:i for i,c in enumerate(vocab)})
            loss = ctc(logp.permute(1,0,2), y.to(args.device), T_input, ylen.to(args.device))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch+1} loss {loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": vocab, "in_ch": in_ch}, "checkpoints/baseline.pt")
    with open("metadata_vocab.json","w") as f: json.dump({"vocab":vocab, "in_ch":in_ch}, f)
    print("saved checkpoints/baseline.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", help="Root dir containing session folders (auto scans)")
    p.add_argument("--train", help="Single session file path to data_train.hdf5")
    p.add_argument("--channels", type=int, default=0, help="Force channel count; 0=infer")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())

