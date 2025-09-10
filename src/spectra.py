"""
spectra.py
----------
Compute and plot turbulence energy spectrum from DNS data.

Usage:
    python src/spectra.py --input examples/data_clean.h5 \
                          --output examples/figures/spectrum.png
"""

import numpy as np
import h5py
import argparse
import os
import matplotlib.pyplot as plt

def compute_energy_spectrum(u):
    """
    Compute 1D isotropic energy spectrum from 3D velocity field u(x,y,z,3).
    Assumes cubic domain and uniform spacing.
    """
    # FFT of each velocity component
    Ux = np.fft.fftn(u[..., 0])
    Uy = np.fft.fftn(u[..., 1])
    Uz = np.fft.fftn(u[..., 2])

    # Energy density in Fourier space
    E_k = (np.abs(Ux)**2 + np.abs(Uy)**2 + np.abs(Uz)**2) / u.size

    # Wavenumber magnitudes
    nx, ny, nz, _ = u.shape
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2).flatten()

    # Bin energy into spherical shells
    E_k = E_k.flatten()
    k_bins = np.arange(0.5, np.max(k_mag)+1, 1.0)
    E_spectrum, _ = np.histogram(k_mag, bins=k_bins, weights=E_k)
    k_center = 0.5 * (k_bins[1:] + k_bins[:-1])

    return k_center, E_spectrum

def plot_spectrum(k, E, output_file=None):
    plt.figure(figsize=(6,4))
    plt.loglog(k, E, label="Energy spectrum")
    # Add reference slope
    ref = (k**(-5/3)) * (E.max() / (k.max()**(-5/3)))
    plt.loglog(k, ref, "--", label="k^-5/3 slope")
    plt.xlabel("Wavenumber k")
    plt.ylabel("E(k)")
    plt.legend()
    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"✅ Saved spectrum plot to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="examples/data_clean.h5",
                        help="Path to DNS dataset (raw or preprocessed)")
    parser.add_argument("--output", type=str, default="examples/figures/spectrum.png",
                        help="Path to save spectrum plot")
    args = parser.parse_args()

    # Load velocity field
    with h5py.File(args.input, "r") as f:
        if "u" in f.keys():
            u = f["u"][...]
        elif "velocity" in f.keys():
            u = f["velocity"][...]
        else:
            raise KeyError("Dataset does not contain 'u' or 'velocity' field.")

    # Compute and plot
    k, E = compute_energy_spectrum(u)
    plot_spectrum(k, E, args.output)
# Fourier-based energy spectra
