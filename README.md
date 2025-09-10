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

## Usage
- Download DNS Data - python src/data_download.py
- Compute Energy Spectrum - python src/spectra.py --input examples/data_sample.h5 --output examples/figures/spectrum.png
- Run SPOD - python src/spod_analysis.py --input examples/data_sample.h5 --outdir results/

---

## Scientific Context

Stratified turbulence appears in geophysical flows, aerospace applications, and energy systems.

DNS provides high-fidelity data, but spectral and modal analyses are key to uncovering physical mechanisms.

This project demonstrates practical ability to:

- Work with DNS data,

- Apply spectral analysis and SPOD,

- Present results in a reproducible open-source format.

---


## Relevant references:

1. Brethouwer, G., et al. (2007). Scaling analysis and simulation of strongly stratified turbulent flows. J. Fluid Mech.
2. He, P., & Basu, S. (2015). DNS of intermittent turbulence in stable stratification. Boundary-Layer Meteorology.
3. Chini, G.P., et al. (2021). Reduced modeling of strongly stratified turbulence. Annu. Rev. Fluid Mech.


