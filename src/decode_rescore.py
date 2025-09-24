# src/decode_rescore.py
import argparse, json, os, torch
from torch.utils.data import DataLoader
from dataset import BT25H5
from train_baseline import EncoderCTC

def lm_load(path):
    ck = torch.load(path, map_location="cpu")
    from lm_char_train import CharLM
    lm = CharLM(ck["vocab"]); lm.load_state_dict(ck["state"]); lm.eval()
    stoi = {c:i for i,c in enumerate(ck["vocab"])}
    return lm, ck["vocab"], stoi

def step_lm_logprob(lm, stoi, prev, nxt):
    # prev/nxt are single chars
    x = torch.tensor([[stoi.get(prev, 0)]])
    with torch.no_grad():
        logits,_ = lm(x)
        logp = torch.log_softmax(logits[:,-1,:], dim=-1)
    return float(logp[0, stoi.get(nxt, 0)])

def collapse_ctc(ids, blank):
    out, prev = [], None
    for i in ids:
        if i==blank: prev=None; continue
        if i!=prev: out.append(i)
        prev=i
    return out

def beam_decode_with_lm(logp, vocab, blank, lm, stoi, beam=8, lm_w=0.5, len_w=0.0):
    # logp: [T,V]
    T,V = logp.shape
    beams = [("", 0.0)]  # (text, score)
    for t in range(T):
        lp = logp[t]  # [V]
        # top-k acoustic
        topk = torch.topk(lp, k=min(16,V)).indices.tolist()
        new=[]
        for text,score in beams:
            for v in topk:
                if v==blank:  # CTC blank: keep text (no char)
                    new.append((text, score + float(lp[v])))
                else:
                    ch = vocab[v]
                    # LM term conditioned on last char (or space)
                    prev = text[-1] if text else " "
                    lm_term = step_lm_logprob(lm, stoi, prev, ch)
                    tot = score + float(lp[v]) + lm_w*lm_term + len_w*len(text)
                    # CTC collapse: avoid repeating the same char unless separated by blank (approx)
                    if not text or ch != text[-1]:
                        new.append((text+ch, tot))
                    else:
                        new.append((text, score + float(lp[v])))  # collapse-ish
        # prune
        new.sort(key=lambda x: x[1], reverse=True)
        beams = new[:beam]
    return beams[0][0]

def main(a):
    # data
    ds = BT25H5(a.val, split=a.split, bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # acoustic model
    ck = torch.load(a.ckpt, map_location="cpu")
    vocab = ck["vocab"]; in_ch = ck["in_ch"]; blank = vocab.index("<BLANK>")
    model = EncoderCTC(in_ch=in_ch, hidden=a.hidden, vocab=vocab)
    model.load_state_dict(ck["state"]); model.eval()

    # LM
    lm, lm_vocab, stoi = lm_load(a.lm_ckpt)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lines = ["id,text"]
    with torch.no_grad():
        for i,(X,_) in enumerate(dl):
            logp = model(X)[0]                # [T,V]
            txt = beam_decode_with_lm(logp, vocab, blank, lm, stoi, beam=a.beam, lm_w=a.lm_weight, len_w=a.len_weight)
            lines.append(f"{i},{txt}")
            if a.max_items and i+1>=a.max_items: break
    with open(a.out, "w") as f: f.write("\n".join(lines))
    print("wrote", a.out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--lm_ckpt", required=True)
    p.add_argument("--out", default="outputs/val_predictions_lm.csv")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--lm_weight", type=float, default=0.5)
    p.add_argument("--len_weight", type=float, default=0.0)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--max_items", type=int, default=0)
    main(p.parse_args())

