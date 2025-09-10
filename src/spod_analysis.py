# SPOD workflow

"""
spod_analysis.py
----------------
Run Spectral Proper Orthogonal Decomposition (SPOD) on DNS dataset.

Usage:
    python src/spod_analysis.py --input examples/data_clean.h5 \
                                --outdir results/spod \
                                --dt 0.01 \
                                --block-size 128
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyspod.spod_low_storage as spod

def run_spod(u, dt=0.01, block_size=128, outdir="results/spod"):
    """
    Run SPOD on DNS velocity dataset.

    Parameters
    ----------
    u : ndarray, shape (nt, nx, ny) or (nt, nx, ny, nz)
        Time series of velocity magnitude or a component
    dt : float
        Time step between snapshots
    block_size : int
        Number of snapshots per block for FFT averaging
    outdir : str
        Output directory for results
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure data is 2D array (time, features)
    nt = u.shape[0]
    data = u.reshape(nt, -1)

    # Initialize SPOD solver
    spod_solver = spod.SPOD_low_storage(
        data=data,
        dt=dt,
        n_snapshots=block_size,
        window='hamming',
        mean_type='blockwise',
        outdir=outdir,
    )

    spod_solver.fit()
    print(f"✅ SPOD completed. Results saved to {outdir}")
    return spod_solver

def plot_first_mode(spod_solver, outdir="results/spod"):
    """Plot first SPOD mode at the most energetic frequency."""
    freqs = spod_solver.freq
    modes_at_freq = spod_solver.modes_at_freq
    # Pick dominant frequency
    idx_max = np.argmax(spod_solver.eigs.mean(axis=1))
    mode = np.real(modes_at_freq[idx_max][:, 0])  # first mode
    mode = mode.reshape(spod_solver.original_shape[1:])

    plt.figure(figsize=(6,4))
    plt.imshow(mode, cmap="RdBu_r")
    plt.colorbar(label="SPOD mode amplitude")
    plt.title(f"First SPOD mode at f = {freqs[idx_max]:.3f}")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "spod_mode.png")
    plt.savefig(fig_path, dpi=300)
    print(f"✅ Saved SPOD mode plot to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="examples/data_clean.h5",
                        help="Path to preprocessed DNS dataset")
    parser.add_argument("--outdir", type=str, default="results/spod",
                        help="Output directory for SPOD results")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step between snapshots")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size for FFT averaging")
    args = parser.parse_args()

    # Load dataset
    with h5py.File(args.input, "r") as f:
        if "u" in f.keys():
            u = f["u"][...]
        else:
            raise KeyError("Dataset does not contain 'u' field.")

    # If dataset has 4D shape (nx,ny,nz,3), reduce to one component
    if u.ndim == 4:
        u = u[..., 0]  # take x-component
    elif u.ndim == 3:
        pass  # already time series
    else:
        raise ValueError("Dataset has unexpected shape.")

    # Add fake time dimension if absent (for demo purposes)
    if u.shape[0] < 10:
        # replicate field to create synthetic time series
        u = np.stack([u for _ in range(64)], axis=0)

    # Run SPOD
    spod_solver = run_spod(u, dt=args.dt, block_size=args.block_size, outdir=args.outdir)

    # Plot first mode
    plot_first_mode(spod_solver, outdir=args.outdir)
