# src/data_download.py
import os
import argparse
import numpy as np
from tqdm import tqdm
from src.io import save_h5

def gen_synthetic(nx=64, ny=64, nz=64, nt=64, dt=0.05, N=2.0, seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)
    z = np.linspace(0, 2*np.pi, nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    u = np.zeros((nt, nz, ny, nx, 3), dtype=np.float32)
    # stratified “pancake” layers + weak inertial oscillations
    for t in tqdm(range(nt), desc="Synth"):
        phase = 0.5*np.sin(N*(t*dt))
        base = np.sin(6*Z) * np.exp(-0.5*np.sin(Z)**2)
        noise = 0.1*rng.standard_normal((nz, ny, nx, 3)).astype(np.float32)
        vort = np.stack([
            np.sin(X + phase) * np.cos(Y) * base,
            -np.cos(X) * np.sin(Y + phase) * base,
            0.1*np.sin(2*Z + phase)
        ], axis=-1).astype(np.float32)
        u[t] = vort + noise
    return u, x, y, z, {"dt": dt, "N": N, "dataset": "synthetic"}

def write_output(out_path, u, x, y, z, attrs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_h5(out_path, u, x, y, z, attrs)

def try_jhtdb_rotstrat(nx, ny, nz, token, t_index=0):
    # Snapshot dataset: no time evolution, returns one time-slice cube
    # For actual cutouts, SciServer Cutout service is recommended
    # Here we provide a placeholder returning None to fall back if not configured
    # See: https://turbulence.pha.jhu.edu/Rotstrat4096.aspx
    return None

def try_jhtdb_time_resolved(nx, ny, nz, nt, token, dataset="isotropic1024coarse"):
    # Time-resolved example via pointwise queries is slow; use SciServer Cutout where possible
    # See: https://turbulence.pha.jhu.edu/help/python/
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="examples/data_sample.h5")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--nt", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--N", type=float, default=2.0)
    parser.add_argument("--use_jhtdb", action="store_true")
    parser.add_argument("--dataset", type=str, default="rotstrat4096")
    parser.add_argument("--token", type=str, default=os.environ.get("JHTDB_TOKEN", ""))
    args = parser.parse_args()

    u = x = y = z = attrs = None
    if args.use_jhtdb and args.token:
        if args.dataset == "rotstrat4096":
            maybe = try_jhtdb_rotstrat(args.nx, args.ny, args.nz, args.token)
            if maybe is not None:
                u, x, y, z, attrs = maybe
        else:
            maybe = try_jhtdb_time_resolved(args.nx, args.ny, args.nz, args.nt, args.token, dataset=args.dataset)
            if maybe is not None:
                u, x, y, z, attrs = maybe

    if u is None:
        u, x, y, z, attrs = gen_synthetic(args.nx, args.ny, args.nz, args.nt, args.dt, args.N)

    write_output(args.output, u, x, y, z, attrs)

if __name__ == "__main__":
    main()
