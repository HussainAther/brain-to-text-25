# src/plot_quicklook.py
import argparse, h5py, numpy as np, matplotlib.pyplot as plt

def main(a):
    with h5py.File(a.path, "r") as f:
        tkeys = sorted([k for k in f.keys() if k.startswith("trial_")])
        if not tkeys:
            raise SystemExit("No trial_* groups found")
        g = f[tkeys[a.trial]]
        arr = None
        for k in ("binned_spikes","neural_data","features","rates"):
            if k in g:
                arr = g[k][...]
                break
        if arr is None:
            # fallback: first 2D dataset
            for k in g.keys():
                if len(g[k].shape) == 2:
                    arr = g[k][...]; break
        if arr is None:
            raise SystemExit("No 2D array found in trial group")

        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # [C,T]
        C, T = arr.shape
        # downsample channels to avoid huge images
        step = max(1, C // 16)
        show = arr[::step, :]

        plt.figure()
        plt.imshow(show, aspect="auto", origin="lower")
        plt.title(f"{a.path} | {tkeys[a.trial]}  (C={C}, T={T})")
        plt.xlabel("time bins"); plt.ylabel("channels (downsampled)")
        if a.out:
            plt.savefig(a.out, dpi=150, bbox_inches="tight")
            print("saved", a.out)
        else:
            plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help=".../data_train.hdf5 or data_val.hdf5")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--out", default="")
    main(p.parse_args())

