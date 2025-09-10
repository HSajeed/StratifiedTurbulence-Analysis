# StratifiedTurbulence-Analysis

This repository demonstrates **post-processing and spectral analysis of DNS turbulence data** using Python.  
It showcases skills in **CFD data handling, turbulence theory, spectral methods, and SPOD** —  
directly relevant to PhD research on **simulations of strongly stratified turbulence**.

---

##  Objectives
- Download and preprocess existing DNS datasets (e.g., from [JHU Turbulence Database](http://turbulence.pha.jhu.edu/)).
- Compute turbulence **energy spectra** and verify scaling laws (e.g., -5/3 inertial range).
- Perform **Spectral Proper Orthogonal Decomposition (SPOD)** using [PySPOD](https://github.com/mendezVKI/PySPOD).
- Visualize **coherent turbulent structures** and compare with literature.
- Provide **clean, reproducible Jupyter notebooks** for each step.

---


## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/StratifiedTurbulence-Analysis.git
cd StratifiedTurbulence-Analysis
pip install -r requirements.txt
```


## Usage
- Download DNS Data
  ```bash
  python src/data_download.py
  ```
- Compute Energy Spectrum
  ```bash
  python src/spectra.py --input examples/data_sample.h5 --output examples/figures/spectrum.png
  ```
- Run SPOD
  ```bash
  python src/spod_analysis.py --input examples/data_sample.h5 --outdir results/
  ```
  ---

### Scientific Context

Stratified turbulence appears in geophysical flows, aerospace applications, and energy systems. DNS provides high-fidelity data, but spectral and modal analyses are key to uncovering physical mechanism.

---


### Relevant references:

- Brethouwer, G., et al. (2007). Scaling analysis and simulation of strongly stratified turbulent flows. J. Fluid Mech.
- He, P., & Basu, S. (2015). DNS of intermittent turbulence in stable stratification. Boundary-Layer Meteorology.
- Chini, G.P., et al. (2021). Reduced modeling of strongly stratified turbulence. Annu. Rev. Fluid Mech.
