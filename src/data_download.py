"""
data_download.py
----------------
Download a sample DNS dataset (HDF5 format) for turbulence analysis.

Default: downloads a coarse isotropic turbulence dataset from JHU Turbulence Database (JHTDB).
"""

import requests
import os

def fetch_dns_data(
    url="https://turbulence.pha.jhu.edu/sample_data/isotropic1024coarse.h5",
    out_file="examples/data_sample.h5"
):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(f"Downloading DNS sample data from:\n{url}\n")
    r = requests.get(url, stream=True)
    with open(out_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✅ Saved dataset to {out_file}")
    return out_file

if __name__ == "__main__":
    fetch_dns_data()
