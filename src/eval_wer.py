# src/eval_wer.py
import argparse, csv, h5py, re
from typing import List, Tuple

# Kaggle: “Words are sequences of characters separated by spaces.
# Punctuation is not evaluated and should not be included.”
_PUNC = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep apostrophes in don't

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = _PUNC.sub("", s)  # drop punctuation like .,!?
    s = re.sub(r"\s+", " ", s)
    return s

def words(s: str) -> List[str]:
    s = normalize(s)
    return [] if not s else s.split(" ")

def load_truth_words(h5_path: str) -> List[List[str]]:
    # loads ground-truth sentences from val file (or train) -> list of word lists
    with h5py.File(h5_path, "r") as f:
        if "sentences" not in f:
            raise RuntimeError(f"No 'sentences' in {h5_path}")
        sents = [s.decode() for s in f["sentences"][:]]
    return [words(x) for x in sents]

def load_pred_words(csv_path: str) -> List[List[str]]:
    pred = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pred.append(words(row["text"]))
    return pred

def edit_distance(ref: List[str], hyp: List[str]) -> int:
    # standard Levenshtein (word-level)
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m]

def aggregate_wer(refs: List[List[str]], hyps: List[List[str]]) -> Tuple[float,int,int]:
    # pad hyps if shorter; extra hyps are ignored for Kaggle-like calc
    L = min(len(refs), len(hyps))
    dist_sum, ref_word_sum = 0, 0
    for i in range(L):
        d = edit_distance(refs[i], hyps[i])
        dist_sum += d
        ref_word_sum += len(refs[i])
    wer = (dist_sum / max(1, ref_word_sum)) * 100.0
    return wer, dist_sum, ref_word_sum

def main(a):
    refs = load_truth_words(a.val)
    hyps = load_pred_words(a.pred)
    wer, dsum, rsum = aggregate_wer(refs, hyps)
    print(f"WER: {wer:.2f}%  (edit_distance_sum={dsum}  total_ref_words={rsum})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True, help="Path to *_val.hdf5 with 'sentences'")
    p.add_argument("--pred", required=True, help="CSV from decode_baseline.py (id,text)")
    main(p.parse_args())

