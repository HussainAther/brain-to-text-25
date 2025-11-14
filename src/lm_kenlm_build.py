# src/lm_kenlm_build.py
import argparse, os, re, h5py, subprocess, shutil

_PUNC = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep apostrophes

def norm(s: str) -> str:
    s = s.strip().lower()
    s = _PUNC.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def gather_sentences(bt25_root, include_val=True):
    files = []
    for d in sorted(os.listdir(bt25_root)):
        p = os.path.join(bt25_root, d)
        if not os.path.isdir(p): continue
        tr = os.path.join(p, "data_train.hdf5")
        if os.path.isfile(tr): files.append(tr)
        if include_val:
            va = os.path.join(p, "data_val.hdf5")
            if os.path.isfile(va): files.append(va)

    sents = []
    for f in files:
        with h5py.File(f, "r") as h:
            if "sentences" in h:
                sents += [norm(s.decode()) for s in h["sentences"][:]]
    return sents

def write_corpus(sentences, out_txt):
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"[kenlm] wrote corpus: {out_txt} ({len(sentences)} lines)")

def run_lmplz(corpus_txt, arpa_out, order=5, memory="4G"):
    if not shutil.which("lmplz"):
        print("[kenlm] WARNING: 'lmplz' not found on PATH. Skipping ARPA build.")
        return False
    cmd = ["lmplz", "-o", str(order), "--memory", memory, "--text", corpus_txt, "--arpa", arpa_out]
    print("[kenlm] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[kenlm] wrote ARPA: {arpa_out}")
    return True

def run_build_binary(arpa, binary_out):
    if not shutil.which("build_binary"):
        print("[kenlm] 'build_binary' not found; skipping binary compilation (ARPA is fine).")
        return False
    cmd = ["build_binary", arpa, binary_out]
    print("[kenlm] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[kenlm] wrote binary: {binary_out}")
    return True

def main(a):
    sents = gather_sentences(a.root, include_val=not a.train_only)
    write_corpus(sents, a.corpus)

    built = False
    if a.build:
        os.makedirs(os.path.dirname(a.arpa) or ".", exist_ok=True)
        built = run_lmplz(a.corpus, a.arpa, order=a.order, memory=a.memory)
        if built and a.binary:
            os.makedirs(os.path.dirname(a.binary) or ".", exist_ok=True)
            run_build_binary(a.arpa, a.binary)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BT25_ROOT (contains t15.YYYY.MM.DD/)")
    p.add_argument("--corpus", default="artifacts/corpus.txt")
    p.add_argument("--arpa", default="artifacts/lm.arpa")
    p.add_argument("--binary", default="artifacts/lm.binary")
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--memory", default="4G")
    p.add_argument("--build", action="store_true", help="Build ARPA (requires lmplz)")
    p.add_argument("--train_only", action="store_true", help="Exclude val splits from the corpus")
    main(p.parse_args())

