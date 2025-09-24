# src/dataset.py
import os, re, glob
import h5py, numpy as np, torch
from torch.utils.data import Dataset

TRIAL_RE = re.compile(r"^trial_\d+$")

def _list_h5(root, split="train"):
    pat = os.path.join(root, "t15.20*/data_%s.hdf5" % split)
    files = sorted(glob.glob(pat))
    if not files:
        flat = os.path.join(root, f"data_{split}.hdf5")
        if os.path.isfile(flat):
            files = [flat]
    if not files:
        raise FileNotFoundError(f"[dataset] No files matching {pat} or {os.path.join(root, f'data_{split}.hdf5')}")
    return files

def _detect_layout(f: h5py.File):
    # 1) 3D/2D at root
    for key in ["neural_data", "binned_spikes", "features", "rates"]:
        if key in f and isinstance(f[key], h5py.Dataset):
            ds = f[key]
            if ds.ndim == 3: return "array3d", {"ds": ds}
            if ds.ndim == 2: return "array2d", {"ds": ds}
    # 2) spike_times at root
    if "spike_times" in f and isinstance(f["spike_times"], h5py.Group):
        return "spike_times", {"grp": f["spike_times"]}
    # 3) many trial_#### groups
    trial_keys = [k for k in f.keys() if TRIAL_RE.match(k) and isinstance(f[k], h5py.Group)]
    if len(trial_keys) >= 1:
        trial_keys.sort()
        return "trial_groups", {"trial_keys": trial_keys}
    return "unknown", {"keys": list(f.keys())}

def _ensure_CT(mat: np.ndarray, n_channels_hint: int = 0):
    """
    Force [C, T] orientation *without* depending strictly on n_channels.
    Heuristics:
      - If one dimension equals a 'plausible channels' count, that dim = C.
      - Otherwise, if one dim >> the other, take larger as T, smaller as C.
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D array per trial, got {mat.shape}")

    h, w = mat.shape
    plausible_channels = {64, 96, 128, 192, 224, 256, 320, 384, 448, 512, 640, 724, 768, 896, 1024}

    # If a hint is provided and matches either dimension, use it
    if n_channels_hint in (h, w) and n_channels_hint > 0:
        return mat if h == n_channels_hint else mat.T

    # If one dimension looks like channels, pick it
    if h in plausible_channels and w not in plausible_channels:
        return mat  # [C, T]
    if w in plausible_channels and h not in plausible_channels:
        return mat.T  # [C, T]

    # If both (or neither) look plausible, assume the *smaller* is channels
    if h <= w:
        return mat  # h=C, w=T
    else:
        return mat.T  # w=C, h=T

class BT25H5(Dataset):
    def __init__(self, path, split="train", n_channels=256, bin_ms=10, max_ms=3000):
        self.n_channels, self.bin_ms, self.max_ms = n_channels, bin_ms, max_ms
        if os.path.isdir(path):
            self.files = _list_h5(path, split=split)
        elif os.path.isfile(path):
            self.files = [path]
        else:
            raise FileNotFoundError(f"[dataset] Not a file or directory: {path}")

        self.handles, self.modes, self.metas = [], [], []
        self.index, self.texts = [], []

        for fi, fp in enumerate(self.files):
            f = h5py.File(fp, "r")
            mode, meta = _detect_layout(f)
            if mode == "unknown":
                raise RuntimeError(f"[dataset] Unknown layout in {fp}. Top keys: {list(f.keys())[:12]}")
            self.handles.append(f); self.modes.append(mode); self.metas.append(meta)

            # sentences if present (some files may not have them)
            n_trials_guess = 0
            if mode == "array3d":
                n_trials_guess = meta["ds"].shape[0]
            elif mode == "array2d":
                n_trials_guess = 1
            elif mode == "spike_times":
                st = meta["grp"]
                n_trials_guess = len(st) if isinstance(st, h5py.Dataset) else len(st.keys())
            elif mode == "trial_groups":
                n_trials_guess = len(meta["trial_keys"])

            if "sentences" in f:
                sents = [s.decode() for s in f["sentences"][:]]
            else:
                sents = [""] * n_trials_guess

            if mode in ("array3d",):
                for i in range(n_trials_guess): self.index.append((fi, i))
                self.texts.extend(sents)
            elif mode in ("array2d",):
                self.index.append((fi, 0)); self.texts.extend(sents or [""])
            elif mode == "spike_times":
                if isinstance(meta["grp"], h5py.Dataset):
                    for i in range(n_trials_guess): self.index.append((fi, i))
                    self.texts.extend(sents)
                else:
                    # group-of-trials
                    for k in sorted(meta["grp"].keys(), key=lambda x: int(x) if x.isdigit() else x):
                        self.index.append((fi, k))
                    if len(sents) < len(self.index):
                        self.texts.extend([""] * (len(self.index) - len(self.texts)))
            elif mode == "trial_groups":
                for tk in meta["trial_keys"]:
                    self.index.append((fi, tk))
                self.texts.extend([""] * len(meta["trial_keys"]))

        if len(self.texts) < len(self.index):
            self.texts.extend([""] * (len(self.index) - len(self.texts)))

    def __len__(self): return len(self.index)

    # ---------- helpers ----------
    def _bin_spike_times_array(self, arr: np.ndarray):
        n_bins = self.max_ms // self.bin_ms
        out = np.zeros((self.n_channels, n_bins), dtype=np.float32)
        if arr.ndim == 1:
            # times only; no channel info -> cannot bin by channel
            # put counts into a dummy channel 0
            b = np.floor_divide(arr, self.bin_ms)
            b = b[(b >= 0) & (b < n_bins)]
            if b.size: np.add.at(out[0], b, 1.0)
            return out
        # if arr looks like [N, 2] (time, channel) or [N, 2] (channel, time)
        if arr.ndim == 2 and arr.shape[1] == 2:
            t = arr[:, 0].astype(np.int64)
            ch = arr[:, 1].astype(int)
            n_bins = self.max_ms // self.bin_ms
            b = np.floor_divide(t, self.bin_ms)
            mask = (b >= 0) & (b < n_bins) & (ch >= 0) & (ch < self.n_channels)
            b, ch = b[mask], ch[mask]
            for c, idxs in _groupby_indices(ch):
                np.add.at(out[c], b[ch == c], 1.0)
            return out
        # otherwise, fall back to empty
        return out

    def _bin_spike_group(self, grp: h5py.Group):
        n_bins = self.max_ms // self.bin_ms
        out = np.zeros((self.n_channels, n_bins), dtype=np.float32)
        # numeric channel members
        chans = [k for k in grp.keys() if k.isdigit()]
        if chans:
            for ks in chans:
                ch = int(ks)
                obj = grp[ks]
                if isinstance(obj, h5py.Dataset):
                    arr = np.asarray(obj[:], dtype=np.int64)
                elif isinstance(obj, h5py.Group):
                    arr = np.asarray(obj.get("times", []), dtype=np.int64)
                else:
                    continue
                if arr.size:
                    b = np.floor_divide(arr, self.bin_ms)
                    b = b[(b >= 0) & (b < n_bins)]
                    if b.size: np.add.at(out[ch], b, 1.0)
            return out
        # look for a common dataset like times/chans stacked
        for key in ["times", "events", "spikes"]:
            if key in grp and isinstance(grp[key], h5py.Dataset):
                return self._bin_spike_times_array(np.asarray(grp[key]))
        return out

    def _load_trial_group(self, f: h5py.File, trial_name: str):
        tg = f[trial_name]
        # A) nested spike_times group
        if "spike_times" in tg and isinstance(tg["spike_times"], h5py.Group):
            return self._bin_spike_group(tg["spike_times"])
        # B) any 2D dataset under common names
        for key in ["neural_data", "binned_spikes", "features", "rates"]:
            if key in tg and isinstance(tg[key], h5py.Dataset):
                return _ensure_CT(np.array(tg[key]))
        # C) per-channel children named "0","1",...
        chans = [k for k in tg.keys() if k.isdigit()]
        if chans:
            # treat as spike-times-per-channel
            n_bins = self.max_ms // self.bin_ms
            out = np.zeros((self.n_channels, n_bins), dtype=np.float32)
            for ks in chans:
                ch = int(ks)
                obj = tg[ks]
                if isinstance(obj, h5py.Dataset):
                    arr = np.asarray(obj[:], dtype=np.int64)
                elif isinstance(obj, h5py.Group):
                    arr = np.asarray(obj.get("times", []), dtype=np.int64)
                else:
                    continue
                if arr.size:
                    b = np.floor_divide(arr, self.bin_ms)
                    b = b[(b >= 0) & (b < n_bins)]
                    if b.size: np.add.at(out[ch], b, 1.0)
            return out
        # D) fall back: first 2D dataset of any name
        for k in tg.keys():
            obj = tg[k]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                return _ensure_CT(np.array(obj[:]))
        # E) last resort: debug info
        raise RuntimeError(f"[dataset] Unrecognized trial layout under {tg.name}. Keys: {list(tg.keys())[:12]}")

    def __getitem__(self, idx):
        fi, ref = self.index[idx]
        f, mode, meta = self.handles[fi], self.modes[fi], self.metas[fi]
        if mode == "array3d":
            X = _ensure_CT(np.array(meta["ds"][ref]))
        elif mode == "array2d":
            X = _ensure_CT(np.array(meta["ds"][:]))
        elif mode == "spike_times":
            st = meta["grp"]
            if isinstance(st, h5py.Dataset):
                X = _ensure_CT(np.array(st[ref]), self.n_channels)
            else:
                grp = st[str(ref)] if isinstance(ref, int) else st[ref]
                X = self._bin_spike_group(grp)
        elif mode == "trial_groups":
            X = self._load_trial_group(f, ref)
        else:
            raise RuntimeError("unreachable mode")
        y = self.texts[idx] if idx < len(self.texts) else ""
        return torch.tensor(X, dtype=torch.float32), y

# small helper
def _groupby_indices(arr):
    # yields (value, mask_indices)
    vals = np.unique(arr)
    for v in vals:
        yield int(v), np.where(arr == v)[0]

