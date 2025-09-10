# src/spectra.py
import argparse
import numpy as np
from numpy.fft import rfftn, fftn, fftshift, ifftshift, rfftfreq, fftfreq
import matplotlib.pyplot as plt
from src.io import load_h5
import os

def _window_3d(shape, kind="hann"):
    if kind is None or kind == "none":
        return np.ones(shape, dtype=np.float32)
    def one(n):
        if kind == "hann":
            return np.hanning(n)
        if kind == "kaiser":
            return np.kaiser(n, beta=14.0)
        return np.hanning(n)
    wz = one(shape)[:, None, None]
    wy = one(shape[21])[None, :, None]
    wx = one(shape[22])[None, None, :]
    w = wz * wy * wx
    return (w / np.sqrt(np.mean(w**2))).astype(np.float32)

def isotropic_spectrum_3d(u):
    # u: [nz, ny, nx, 3], returns k, E(k)
    nz, ny, nx, _ = u.shape
    w = _window_3d((nz, ny, nx))
    uhat = np.zeros((nz, ny, nx//2+1), dtype=np.float64)
    for c in range(u.shape[-1]):
        Uc = rfftn(u[..., c] * w, axes=(0,1,2))
        uhat += (np.abs(Uc)**2)
    kx = rfftfreq(nx) * nx
    ky = fftfreq(ny) * ny
    kz = fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='xy')
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    kk = kk.transpose(2,1,0)  # align to rfftn output indexing
    kmax = int(np.max(kk))
    bins = np.arange(0, kmax+1)
    E = np.zeros(kmax+1, dtype=np.float64)
    counts = np.zeros(kmax+1, dtype=np.int64)
    flat_k = kk.ravel().astype(int)
    flat_e = uhat.ravel()
    np.add.at(E, flat_k, flat_e)
    np.add.at(counts, flat_k, 1)
    mask = counts > 0
    E[mask] /= counts[mask]
    return bins[mask], E[mask]

def horizontal_spectrum(u):
    # Average over z to form layers, 2D FFT in x–y → kh-binned spectrum
    nz, ny, nx, c = u.shape
    w2 = _window_3d((1, ny, nx))
    e_acc = None
    for iz in range(nz):
        layer = u[iz]
        Eh = np.zeros((ny, nx//2+1), dtype=np.float64)
        for comp in range(c):
            Uh = rfftn(layer[..., comp] * w2, axes=(0,1))
            Eh += np.abs(Uh)**2
        e_acc = Eh if e_acc is None else e_acc + Eh
    e_acc /= nz
    kx = rfftfreq(nx) * nx
    ky = fftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    kh = np.sqrt(KX**2 + KY**2)
    kh = kh
    kmax = int(np.max(kh))
    bins = np.arange(0, kmax+1)
    E = np.zeros(kmax+1)
    counts = np.zeros(kmax+1, dtype=np.int64)
    flat_k = kh.ravel().astype(int)
    flat_e = e_acc.ravel()
    np.add.at(E, flat_k, flat_e)
    np.add.at(counts, flat_k, 1)
    mask = counts > 0
    E[mask] /= counts[mask]
    return bins[mask], E[mask]

def vertical_spectrum(u):
    # Average over x–y planes, 1D FFT along z → kz spectrum
    nz, ny, nx, c = u.shape
    w = np.hanning(nz).astype(np.float32)
    e_acc = np.zeros(nz, dtype=np.float64)
    for jy in range(ny):
        for ix in range(nx):
            col = u[:, jy, ix, :]  # [nz, c]
            for comp in range(c):
                Uz = np.fft.fft(col[:, comp] * w)
                e_acc += np.abs(Uz)**2
    e_acc /= (nx * ny)
    kz = np.fft.fftfreq(nz) * nz
    kz = np.abs(kz.astype(int))
    kmax = int(np.max(kz))
    E = np.zeros(kmax+1)
    counts = np.zeros(kmax+1, dtype=np.int64)
    np.add.at(E, kz, e_acc.astype(np.float64))
    np.add.at(counts, kz, 1)
    mask = counts > 0
    E[mask] /= counts[mask]
    return np.arange(0, kmax+1)[mask], E[mask]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, default="examples/figures/spectrum.png")
    ap.add_argument("--mode", choices=["isotropic", "horizontal", "vertical"], default="isotropic")
    ap.add_argument("--t_index", type=int, default=0)
    args = ap.parse_args()

    u, x, y, z, attrs = load_h5(args.input)
    field = u[args.t_index]  # [z, y, x, 3]

    if args.mode == "isotropic":
        k, E = isotropic_spectrum_3d(field)
        label = "E(k)"
    elif args.mode == "horizontal":
        k, E = horizontal_spectrum(field)
        label = "E(k_h)"
    else:
        k, E = vertical_spectrum(field)
        label = "E(k_z)"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.loglog(k[1:], E[1:]+1e-16, label=label)
    plt.xlabel("wavenumber")
    plt.ylabel("energy")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)

if __name__ == "__main__":
    main()
