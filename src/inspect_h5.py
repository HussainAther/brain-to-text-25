# src/inspect_h5.py
import argparse, h5py, itertools

def walk(f, max_kids=20):
    def _walk(g, indent=0):
        if isinstance(g, h5py.Dataset):
            print("  "*indent + f"- {g.name}  shape={g.shape} dtype={g.dtype}")
        else:
            kids = list(g.keys())
            print("  "*indent + f"[{g.name}] children={len(kids)}")
            for k in kids[:max_kids]:
                _walk(g[k], indent+1)
            if len(kids) > max_kids:
                print("  "*(indent+1) + f"...(+{len(kids)-max_kids} more)")
    _walk(f)

def sample_sentence(f):
    if "sentences" in f:
        s = f["sentences"][0].decode()
        print("example sentence:", s)

def main(a):
    with h5py.File(a.path, "r") as f:
        walk(f, max_kids=a.max)
        sample_sentence(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--max", type=int, default=40)
    main(p.parse_args())

