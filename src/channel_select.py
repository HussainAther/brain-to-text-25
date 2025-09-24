# src/channel_select.py
import argparse, h5py, numpy as np, os

def channel_stats(h5_path, bin_key_order=("binned_spikes","neural_data","features","rates")):
    with h5py.File(h5_path,"r") as f:
        # try root trial groups -> first 32 trials for stats
        trials = [k for k in f.keys() if k.startswith("trial_")]
        chans_sum=None; chans_sq=None; nT=0; C=None
        for tk in trials[:32]:
            g=f[tk]
            arr=None
            for k in bin_key_order:
                if k in g: arr = np.array(g[k]); break
            if arr is None: continue
            if arr.shape[0] < arr.shape[1]: arr = arr.T  # [C,T]
            if C is None: C = arr.shape[0]
            arr = arr[:C, :]
            if chans_sum is None:
                chans_sum = arr.sum(axis=1)
                chans_sq  = (arr**2).sum(axis=1)
            else:
                chans_sum += arr.sum(axis=1)
                chans_sq  += (arr**2).sum(axis=1)
            nT += arr.shape[1]
        mean = chans_sum / max(1,nT)
        var  = chans_sq / max(1,nT) - mean**2
        return mean, var

def main(a):
    mean, var = channel_stats(a.h5)
    snr = np.where(var>0, mean/np.sqrt(var), 0.0)
    idx = np.argsort(-snr)  # high to low
    keep = idx[:a.keep]
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/channels_keep.npy", keep)
    print("saved artifacts/channels_keep.npy", keep.shape)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--keep", type=int, default=256)
    main(p.parse_args())

