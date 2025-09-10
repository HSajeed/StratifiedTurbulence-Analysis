# src/io.py
import h5py
import numpy as np
from typing import Tuple, Dict, Any

def save_h5(path: str, u: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, attrs: Dict[str, Any] = None):
    with h5py.File(path, "w") as f:
        f.create_dataset("u", data=u, compression="gzip", compression_opts=4)
        f.create_dataset("coords/x", data=x)
        f.create_dataset("coords/y", data=y)
        f.create_dataset("coords/z", data=z)
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v

def load_h5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        u = f["u"][...]
        x = f["coords/x"][...]
        y = f["coords/y"][...]
        z = f["coords/z"][...]
        attrs = dict(f.attrs.items())
    assert u.ndim == 5 and u.shape[-1] in (2, 3), "u must be [t, z, y, x, c]"  # 2D/3D vars
    return u, x, y, z, attrs
