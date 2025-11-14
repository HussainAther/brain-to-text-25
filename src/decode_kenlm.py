# src/decode_kenlm.py
import argparse, os, csv, math, torch, kenlm
from torch.utils.data import DataLoader
from dataset import BT25H5
from train_baseline import EncoderCTC
from utils import device_str

def logsumexp(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))

def prefix_beam_search_ctc_lm(logp, vocab, blank_idx, lm, alpha=0.8, beta=0.0, len_bonus=0.0, beam_width=12, topk=16):
    """
    logp: [T, V] (torch) log-probs over vocab at each time step (already log_softmaxed)
    vocab: list of symbols (first is <BLANK>)
    LM: kenlm.Model (word-level). We apply LM over the entire string; delta score = LM(prefix+ch)-LM(prefix).
    Implements character-level prefix beam search with CTC and KenLM rescoring.
    """
    T, V = logp.shape
    # Beam: map prefix string -> (pb, pnb, lm_score, last_char) in log domain
    # pb: prob of prefix ending with blank; pnb: prob ending with non-blank
    beams = {"": (-0.0, -math.inf, 0.0, None)}  # log probs; start with empty

    for t in range(T):
        lp = logp[t]
        candidates = torch.topk(lp, k=min(V, topk))
        cand_ids = candidates.indices.tolist()
        cand_vals = candidates.values.tolist()

        new_beams = {}
        def update(prefix, pb, pnb, lm_s, last_c):
            # merge to new_beams with log-sum-exp on both pb and pnb
            if prefix in new_beams:
                (pb0, pnb0, lm0, lc0) = new_beams[prefix]
                new_beams[prefix] = (logsumexp(pb0, pb), logsumexp(pnb0, pnb), lm0, lc0)
            else:
                new_beams[prefix] = (pb, pnb, lm_s, last_c)

        for prefix, (pb, pnb, lm_s, last_c) in beams.items():
            # 1) Extend with blank
            p_blank = lp[blank_idx].item()
            update(prefix, pb + p_blank, pnb + p_blank, lm_s, last_c)

            # 2) Extend with characters
            for vi, logv in zip(cand_ids, cand_vals):
                if vi == blank_idx: 
                    continue
                ch = vocab[vi]

                # CTC merge rule: if last char equals ch, only come from blank
                if last_c == ch:
                    pnb_new = pnb  # stay same char without passing through blank is disallowed here
                    # Only path via blank -> new non-blank with same char
                    pnb_new = logsumexp(pnb_new, pb + logv)
                    # LM delta (we apply LM on char addition)
                    # Word-level LM over characters: approximate with total score difference
                    # This is simpler than stateful; good enough for char decoding.
                    lm_new = lm_s + (lm.score(prefix + ch, bos=False, eos=False) - lm.score(prefix, bos=False, eos=False))
                    prefix_new = prefix + ch
                    update(prefix_new, -math.inf, pnb_new + alpha * lm_new + beta * (1.0 if ch == " " else 0.0) + len_bonus, lm_new, ch)
                else:
                    # From blank or different last char
                    pnb_new = logsumexp(-math.inf, pb + logv)
                    pnb_new = logsumexp(pnb_new, pnb + logv)
                    lm_new = lm_s + (lm.score(prefix + ch, bos=False, eos=False) - lm.score(prefix, bos=False, eos=False))
                    prefix_new = prefix + ch
                    update(prefix_new, -math.inf, pnb_new + alpha * lm_new + beta * (1.0 if ch == " " else 0.0) + len_bonus, lm_new, ch)

        # prune
        def total_score(state):
            pb, pnb, lm_s, _ = state
            return logsumexp(pb, pnb)
        # Keep top beam_width by total score
        beams = dict(sorted(new_beams.items(), key=lambda kv: total_score(kv[1]), reverse=True)[:beam_width])

    # Get best
    best = max(beams.items(), key=lambda kv: logsumexp(kv[1][0], kv[1][1]))[0]
    # Basic cleanup: collapse repeated characters that survived incorrectly (rare)
    # and trim whitespace
    return " ".join(best.strip().split())

def main(a):
    dev = torch.device(device_str())
    ds = BT25H5(a.val, split=a.split, bin_ms=a.bin_ms, max_ms=a.max_ms)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # Acoustic model
    ck = torch.load(a.ckpt, map_location=dev)
    vocab = ck["vocab"]; in_ch = ck.get("in_ch", ds[0][0].shape[0])
    blank = vocab.index("<BLANK>")
    model = EncoderCTC(in_ch=in_ch, hidden=a.hidden, vocab=vocab).to(dev)
    model.load_state_dict(ck["state"]); model.eval()

    # KenLM (can load ARPA or binary)
    lm_path = a.lm if a.lm else "artifacts/lm.binary"
    if not os.path.exists(lm_path):
        lm_path = "artifacts/lm.arpa"
    if not os.path.exists(lm_path):
        raise SystemExit(f"KenLM model not found: {a.lm or 'artifacts/lm.(binary|arpa)'}")
    lm = kenlm.Model(lm_path)
    print(f"[kenlm] loaded: {lm_path}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","text"])
        with torch.no_grad():
            for i, (X, _) in enumerate(dl):
                X = X.to(dev)
                logp = model(X)[0]  # [T,V]
                txt = prefix_beam_search_ctc_lm(
                    logp.cpu(), vocab, blank, lm,
                    alpha=a.lm_weight, beta=a.word_bonus, len_bonus=a.len_weight,
                    beam_width=a.beam, topk=a.topk
                )
                w.writerow([i, txt])
                if a.max_items and (i+1) >= a.max_items: break
    print("wrote", a.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True, help=".../data_val.hdf5 (or test/train with --split)")
    p.add_argument("--split", default="val", choices=["val","test","train"])
    p.add_argument("--ckpt", required=True, help="Acoustic checkpoint (.pt)")
    p.add_argument("--lm", default="", help="Path to KenLM .binary or .arpa (default: artifacts/lm.*)")
    p.add_argument("--out", default="outputs/val_kenlm.csv")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beam", type=int, default=12)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--lm_weight", type=float, default=0.8)
    p.add_argument("--word_bonus", type=float, default=0.0)
    p.add_argument("--len_weight", type=float, default=0.0)
    p.add_argument("--bin_ms", type=int, default=10)
    p.add_argument("--max_ms", type=int, default=3000)
    p.add_argument("--max_items", type=int, default=0)
    main(p.parse_args())

