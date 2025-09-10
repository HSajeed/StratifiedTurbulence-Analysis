"""
preprocessing.py
----------------
Preprocess DNS dataset:
- Normalize velocity fields
- Optionally downsample to smaller grid
- Save cleaned dataset to HDF5

Usage:
    python src/preprocessing.py --input examples/data_sample.h5 \
                                --output examples/data_clean.h5 \
                                --downsample 2
"""

import h5py
import numpy as np
import argparse
import os

def preprocess_data(u, downsample=1, normalize=True):
    """
    Preprocess velocity field array.
    Parameters
    ----------
    u : ndarray, shape (nx, ny, nz, 3)
        Velocity field
    downsample : int
        Factor to downsample (e.g., 2 -> half resolution)
    normalize : bool
        If True, zero-mean and unit-variance normalization
    """
    # Downsample
    if downsample > 1:
        u = u[::downsample, ::downsample, ::downsample, :]

    # Normalize
    if normalize:
        mean = np.mean(u, axis=(0,1,2), keepdims=True)
        std = np.std(u, axis=(0,1,2), keepdims=True) + 1e-12
        u = (u - mean) / std

    return u

def save_data(u, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("u", data=u)
    print(f"✅ Saved preprocessed data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="examples/data_sample.h5",
                        help="Path to raw DNS dataset")
    parser.add_argument("--output", type=str, default="examples/data_clean.h5",
                        help="Path to save cleaned dataset")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsampling factor (default: 1 = no downsampling)")
    args = parser.parse_args()

    # Load velocity field
    with h5py.File(args.input, "r") as f:
        if "u" in f.keys():
            u = f["u"][...]
        elif "velocity" in f.keys():
            u = f["velocity"][...]
        else:
            raise KeyError("Dataset does not contain 'u' or 'velocity' field.")

    # Preprocess
    u_clean = preprocess_data(u, downsample=args.downsample, normalize=True)

    # Save
    save_data(u_clean, args.output)

