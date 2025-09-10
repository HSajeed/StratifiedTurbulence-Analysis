# src/config.py
from dataclasses import dataclass

@dataclass
class DataConfig:
    input_h5: str = "examples/data_sample.h5"
    output_dir: str = "results"
    dataset_name: str = "synthetic"  # "rotstrat4096", "isotropic1024", etc.
    # Physical metadata (useful for reporting/regime IDs)
    N: float | None = None    # Brunt–Väisälä frequency [1/s]
    dt: float | None = None   # time step [s]

@dataclass
class SpectraConfig:
    window: str = "hann"
    detrend: str | None = None
    pad_to_pow2: bool = True
    isotropic_bins: int = 128
    horizontal_bins: int = 128
    vertical_bins: int = 128

@dataclass
class SPODConfig:
    block_size: int = 128
    overlap: int = 64
    window: str = "hann"
    normalize_weights: bool = True
    variables: int = 3  # velocity components
