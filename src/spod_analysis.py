# spod_analysis

import argparse
import os
import numpy as np
from src.io import load_h5
from pyspod.spod.standard import Standard as SPOD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results/spod")
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--window", type=str, default="hann")
    ap.add_argument("--vars", type=int, default=3)
    ap.add_argument("--max_snapshots", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    u, x, y, z, attrs = load_h5(args.input)
    dt = float(attrs.get("dt", 1.0))
    T, Z, Y, X, C = u.shape
    if args.max_snapshots and args.max_snapshots < T:
        u = u[:args.max_snapshots]

    # reshape for PySPOD: [time, space, vars]
    space = Z * Y * X
    data = u.reshape(u.shape, space, C)  # time, space, vars

    params = {
        "time_step": dt,
        "n_space_dims": 3,
        "n_variables": C,
        "n_dft": args.block_size,
        "overlap": args.overlap,
        "mean_type": "block",         # robust against slow drifts
        "normalize": True,            # weight by variance
        "save_dir": args.outdir,
        "weights": None,
        "dtype": "double",
        "window_type": args.window,
    }
    os.makedirs(args.outdir, exist_ok=True)
    spod = SPOD(params=params)
    spod.fit(data)

    # Example: load leading modes and compute coefficients at a chosen frequency
    freqs = np.load(os.path.join(args.outdir, "freq.npy"))
    evals = np.load(os.path.join(args.outdir, "eigvals.npy"))

    # Optional coefficients via convolution or oblique projection (utils API)
    # from pyspod.spod.utils import compute_coeffs_conv, compute_coeffs_op
    # coeffs_dir, _ = compute_coeffs_conv(data, args.outdir, modes_idx=, freq_idx=[21])

if __name__ == "__main__":
    main()
