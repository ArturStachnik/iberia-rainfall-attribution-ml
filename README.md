# Iberia Rainfall Attribution

**Machine-learning attribution of synoptic-scale climate variability on rainfall isotopic composition in southeastern Iberia.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19075045.svg)](https://doi.org/10.5281/zenodo.19075045)
[![Preprint](https://img.shields.io/badge/preprint-EGUsphere-blue.svg)](https://doi.org/10.5194/egusphere-2026-2058)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

This repository quantifies the **seasonal hierarchy of synoptic controls on rainfall isotopic composition** (δ¹⁸O, δ²H, d-excess) in southeastern Iberia. It integrates a regional precipitation-isotope monitoring network (Sierra de Segura) with two leading atmospheric circulation indices — the **North Atlantic Oscillation (NAO)** and the **Western Mediterranean Oscillation (WeMO)** — and Lagrangian air-mass trajectories to identify which large-scale processes drive isotopic signatures month by month.

Random Forest with SHAP interpretability is used to extract the seasonal hierarchy of drivers; HYSPLIT trajectory clustering links those drivers to physical transport regimes.

This is the code companion to:

> Stachnik, A., Gázquez, F., Lope Morales González, A., Morellón, M., González Ramón, A., Moral Martos, F., Jiménez Espinosa, R., Martín-Chivelet, J. *Influence of synoptic patterns (NAO vs. WeMO) on rainfall isotopic composition in SE Iberia: A machine learning approach.* Preprint at EGUsphere ([10.5194/egusphere-2026-2058](https://doi.org/10.5194/egusphere-2026-2058)), under review at *Weather and Climate Dynamics*.

## Highlights

- End-to-end reproducible pipeline from raw monthly isotope records to publication figures.
- Random Forest + SHAP-based attribution of seasonal δ¹⁸O variability.
- HYSPLIT backward-trajectory post-processing and unsupervised clustering of transport regimes.
- WeMO index reconstruction harmonised with NAO.
- Conda-based environment specification for full reproducibility.

## Methods

| Step | Approach |
|---|---|
| Isotope preprocessing | Monthly aggregation, harmonisation across stations |
| Circulation indices | NAO (NOAA/CPC), WeMO (reconstructed in-repo) |
| Attribution model | Random Forest regression |
| Interpretability | SHAP values + permutation importance |
| Transport diagnostics | HYSPLIT backward trajectories, unsupervised clustering |
| Reproducibility | Conda environment, scripted figure generation |

## Repository structure

```
.
├── data/                  # processed datasets, derived products
├── notebooks/             # exploratory and figure-generation notebooks
├── src/                   # core pipeline modules
│   └── wemo/              # WeMO index reconstruction
├── trajectories/          # HYSPLIT post-processing
├── figures/               # publication figures (auto-generated)
├── environment.yml        # Conda specification
└── README.md
```

## Quickstart

```bash
git clone https://github.com/<your-username>/iberia-rainfall-attribution
cd iberia-rainfall-attribution
conda env create -f environment.yml
conda activate iberia-attribution
python src/run_pipeline.py
```

To regenerate manuscript figures only:

```bash
python src/figures.py --output figures/
```

## Data sources

| Dataset | Source |
|---|---|
| Precipitation isotopes | Sierra de Segura monitoring network (in-repo) |
| NAO index | [NOAA Climate Prediction Center](https://www.cpc.ncep.noaa.gov/) |
| WeMO index | Reconstructed in this work; see `src/wemo/` |
| Air-mass trajectories | [HYSPLIT (NOAA ARL)](https://www.ready.noaa.gov/HYSPLIT.php) |
| Station metadata | AEMET, ARPAV |

External datasets are subject to their original providers' terms of use.

## Citation

If you use this code, please cite:

```bibtex
@software{stachnik_iberia_2026,
  author       = {Stachnik, Artur and Gázquez, Fernando and Lope Morales González, Antonio and
                  Morellón, Mario and González Ramón, Antonio and Moral Martos, Francisco and
                  Jiménez Espinosa, Rosario and Martín-Chivelet, Javier},
  title        = {Iberia Rainfall Attribution: ML-based attribution of synoptic controls on
                  rainfall isotopic composition in SE Iberia},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.19075045}
}
```

## License

Code released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). External datasets remain subject to their original licenses.

## Acknowledgments

This work was supported by an **FPI Predoctoral Research Fellowship** (Spanish State Research Agency, AEI) at Universidad Complutense de Madrid.
