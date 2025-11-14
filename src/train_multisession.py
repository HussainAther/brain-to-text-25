# src/train_multisession.py
import argparse, os, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import BT25H5
from train_baseline import EncoderCTC, make_vocab, encode_targets, make_collate
from utils import set_seed, device_str, SmoothedValue, Timer, save_ckpt

def build_vocab_from_roots(root_dirs, sample_per_ds=128):
    texts = []
    for root in root_dirs:
        ds = BT25H5(root, split="train", n_channels=0)
        for i in range(min(sample_per_ds, len(ds))):
            _, t = ds[i]
            texts.append(t)
    vocab, _ = make_vocab(texts)
    return vocab

def build_concat_ds(root_dirs, split="train", n_channels=0, bin_ms=10, max_ms=3000):
    datasets = [BT25H5(r, split=split, n_channels=n_channels, bin_ms=bin_ms, max_ms=max_ms) for r in root_dirs]
    # filter out empties
    datasets = [d for d in datasets if len(d) > 0]
    if not datasets:
        raise SystemExit("No datasets found. Check paths.")
    return ConcatDataset(datasets)

def main(a):
    set_seed(a.seed)
    dev = torch.device(a.device or device_str())
    roots = [os.path.join(a.root, d) for d in sorted(os.listdir(a.root)) if d.startswith("t15.")]
    print(f"[data] sessions found: {len(roots)}")

    vocab = build_vocab_from_roots(roots, sample_per_ds=a.vocab_samples)
    blank = vocab.index("<BLANK>")
    print(f"[vocab] size={len(vocab)}")

    ds = build_concat_ds(roots, split="train", n_channels=a.channels, bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(
        ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
        pin_memory=(dev.type == "cuda"), collate_fn=make_collate(target_C=a.channels or None),
        drop_last=False
    )
    model = EncoderCTC(in_ch=(a.channels or 256), hidden=a.hidden, vocab=vocab).to(dev)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    meter = SmoothedValue(); timer = Timer()
    step = 0
    for ep in range(1, a.epochs+1):
        model.train()
        for X, texts, tlen in dl:
            X = X.to(dev)
            logp = model(X) # [B,T,V]
            y, ylen = encode_targets(texts, {c:i for i,c in enumerate(vocab)})
            loss = ctc(logp.permute(1,0,2), y.to(dev), tlen.to(dev), ylen.to(dev))

            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()

            step += 1; meter.update(loss.item())
            if step % a.log_every == 0:
                print(f"[train] ep {ep} step {step} loss {meter.avg:.4f} B={X.size(0)} C={X.size(1)} T≈{int(tlen.float().mean())} t={timer.lap():.1f}s")
            if a.max_batches and step >= a.max_batches:
                break
        if a.save_every and (ep % a.save_every == 0):
            save_ckpt({"state": model.state_dict(), "vocab": vocab, "in_ch": (a.channels or 256)},
                      os.path.join(a.ckpt_dir, f"baseline_ep{ep}.pt"))
        if a.max_batches and step >= a.max_batches:
            break

    os.makedirs(a.ckpt_dir, exist_ok=True)
    save_ckpt({"state": model.state_dict(), "vocab": vocab, "in_ch": (a.channels or 256)},
              os.path.join(a.ckpt_dir, "baseline_multisession.pt"))
    print("[done] saved", os.path.join(a.ckpt_dir, "baseline_multisession.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BT25_ROOT dir containing t15.YYYY.MM.DD/")
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--vocab_samples", type=int, default=256)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    main(p.parse_args())
# src/train_multisession.py
import argparse, os, random, torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import BT25H5
from train_baseline import EncoderCTC, make_vocab, encode_targets, make_collate
from utils import set_seed, device_str, SmoothedValue, Timer, save_ckpt

def build_vocab_from_roots(root_dirs, sample_per_ds=128):
    texts = []
    for root in root_dirs:
        ds = BT25H5(root, split="train", n_channels=0)
        for i in range(min(sample_per_ds, len(ds))):
            _, t = ds[i]
            texts.append(t)
    vocab, _ = make_vocab(texts)
    return vocab

def build_concat_ds(root_dirs, split="train", n_channels=0, bin_ms=10, max_ms=3000):
    datasets = [BT25H5(r, split=split, n_channels=n_channels, bin_ms=bin_ms, max_ms=max_ms) for r in root_dirs]
    # filter out empties
    datasets = [d for d in datasets if len(d) > 0]
    if not datasets:
        raise SystemExit("No datasets found. Check paths.")
    return ConcatDataset(datasets)

def main(a):
    set_seed(a.seed)
    dev = torch.device(a.device or device_str())
    roots = [os.path.join(a.root, d) for d in sorted(os.listdir(a.root)) if d.startswith("t15.")]
    print(f"[data] sessions found: {len(roots)}")

    vocab = build_vocab_from_roots(roots, sample_per_ds=a.vocab_samples)
    blank = vocab.index("<BLANK>")
    print(f"[vocab] size={len(vocab)}")

    ds = build_concat_ds(roots, split="train", n_channels=a.channels, bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(
        ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
        pin_memory=(dev.type == "cuda"), collate_fn=make_collate(target_C=a.channels or None),
        drop_last=False
    )
    model = EncoderCTC(in_ch=(a.channels or 256), hidden=a.hidden, vocab=vocab).to(dev)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    meter = SmoothedValue(); timer = Timer()
    step = 0
    for ep in range(1, a.epochs+1):
        model.train()
        for X, texts, tlen in dl:
            X = X.to(dev)
            logp = model(X) # [B,T,V]
            y, ylen = encode_targets(texts, {c:i for i,c in enumerate(vocab)})
            loss = ctc(logp.permute(1,0,2), y.to(dev), tlen.to(dev), ylen.to(dev))

            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()

            step += 1; meter.update(loss.item())
            if step % a.log_every == 0:
                print(f"[train] ep {ep} step {step} loss {meter.avg:.4f} B={X.size(0)} C={X.size(1)} T≈{int(tlen.float().mean())} t={timer.lap():.1f}s")
            if a.max_batches and step >= a.max_batches:
                break
        if a.save_every and (ep % a.save_every == 0):
            save_ckpt({"state": model.state_dict(), "vocab": vocab, "in_ch": (a.channels or 256)},
                      os.path.join(a.ckpt_dir, f"baseline_ep{ep}.pt"))
        if a.max_batches and step >= a.max_batches:
            break

    os.makedirs(a.ckpt_dir, exist_ok=True)
    save_ckpt({"state": model.state_dict(), "vocab": vocab, "in_ch": (a.channels or 256)},
              os.path.join(a.ckpt_dir, "baseline_multisession.pt"))
    print("[done] saved", os.path.join(a.ckpt_dir, "baseline_multisession.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BT25_ROOT dir containing t15.YYYY.MM.DD/")
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--vocab_samples", type=int, default=256)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    main(p.parse_args())

