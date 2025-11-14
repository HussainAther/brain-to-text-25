# src/utils.py
import os, re, json, random, math, time
import torch

_PUNC = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep apostrophes

def set_seed(s=1337):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _PUNC.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def ctc_collapse(ids, blank_id):
    out, prev = [], None
    for i in ids:
        if i == blank_id:
            prev = None
            continue
        if i != prev:
            out.append(i)
        prev = i
    return out

def save_ckpt(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)

class SmoothedValue:
    def __init__(self):
        self.n = 0; self.total = 0.0
    def update(self, v):
        self.n += 1; self.total += float(v)
    @property
    def avg(self):
        return self.total / max(1,self.n)

class Timer:
    def __init__(self): self.t0 = time.time()
    def lap(self): return time.time() - self.t0

