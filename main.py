import os
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
COMP_NAME = "brain-to-text-25"

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not any(os.scandir(DATA_DIR)):
        print(f"Downloading {COMP_NAME} dataset from Kaggle...")
        subprocess.run(["pip", "install", "-q", "kaggle"])
        subprocess.run(["kaggle", "competitions", "download", "-c", COMP_NAME, "-p", DATA_DIR])
        subprocess.run(["unzip", "-o", f"{DATA_DIR}/{COMP_NAME}.zip", "-d", DATA_DIR])
    else:
        print("Dataset already present. Skipping download.")

def preview_hdf5():
    # Search for any .h5 or .hdf5 file
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith((".h5", ".hdf5")):
                file_path = os.path.join(root, file)
                print(f"Opening {file_path}...")
                with h5py.File(file_path, 'r') as f:
                    print("Keys in file:", list(f.keys()))
                    for key in f.keys():
                        try:
                            arr = np.array(f[key])
                            print(f"{key}: shape {arr.shape}, dtype {arr.dtype}")
                            if arr.ndim == 2 and arr.shape[0] < 1000:
                                plt.imshow(arr, aspect='auto', cmap='viridis')
                                plt.colorbar()
                                plt.title(f"{key} preview")
                                os.makedirs("outputs", exist_ok=True)
                                plt.savefig(f"outputs/{key}_heatmap.png")
                                plt.close()
                        except Exception as e:
                            print(f"Could not load {key}: {e}")
                return

if __name__ == "__main__":
    download_data()
    preview_hdf5()
