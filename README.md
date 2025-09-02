# forest-age-upscale

[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL--1.2-blue.svg)](#license)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-brightgreen)
[![Dataset DOI](https://img.shields.io/badge/GAMI%20Dataset-10.5880%2FGFZ.1.4.2023.006-orange)](https://doi.org/10.5880/GFZ.1.4.2023.006)
[![Paper](https://img.shields.io/badge/Nat%20Ecol%20Evol-10.1038%2Fs41559--025--02821--5-purple)](https://www.nature.com/articles/s41559-025-02821-5)

**forest-age-upscale** is a Python package to upscale forest age from heterogeneous observations using a machine-learning and biophysical fusion pipeline.  
It also provides tools to access and process the **GAMI (Global Age Mapping Integration)** products.

The accompanying paper is:  
**“Global covariation of forest age transitions with the net carbon balance”**  
*Nature Ecology & Evolution* (2025). [DOI: 10.1038/s41559-025-02821-5](https://www.nature.com/articles/s41559-025-02821-5)

---

## Table of contents
- [What’s inside](#whats-inside)
- [Install](#install)
- [Quick start](#quick-start)
- [Using GAMI data](#using-gami-data)
- [Reproducing paper figures](#reproducing-paper-figures)
- [Repository layout](#repository-layout)
- [Cite](#cite)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## What’s inside
- **Upscaling pipeline** (`ageUpscaling/`): core modules for loading data, applying transformations, training models, and generating upscaled forest-age cubes.
- **Analysis** (`analysis/`): scripts to reproduce the figures and tables in the paper.

The package is tightly linked to the **GAMI v* dataset** — a global 100 m forest-age ensemble for 2010 and 2020 with quantified uncertainty.

---

## Install
Requirements: **Python ≥ 3.11**

```bash
# clone
git clone https://github.com/simonbesnard1/forest-age-upscale.git
cd forest-age-upscale

# editable install
pip install -e .
```

> It’s recommended to use a clean virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).

---

## Using GAMI data

**Dataset:** Global Age Mapping Integration (GAMI) (100 m, 2010 & 2020; 20-member ensemble with uncertainty).  
**DOI:** [10.5880/GFZ.1.4.2023.006](https://doi.org/10.5880/GFZ.1.4.2023.006)

**Load with xarray:**

```python
import xarray as xr

ds = xr.open_dataset("/path/to/GAMI_v2.0/age_2010_2020_100m.nc")
print(ds)
```
---

## Reproducing paper figures

The folder `analysis/generate_figures/` contains scripts to reproduce all figures from the paper.  
Each script specifies required inputs in its header. Data are provided via the [Nextcloud folder](https://nextcloud.gfz.de/s/Kx4qyBMFqXtQLDD).

---

## Repository layout

```
├─ ageUpscaling/                # Core package
│  ├─ core/
│  ├─ cubegen/
│  ├─ dataloaders/
│  ├─ diagnostic/
│  ├─ fidc_cube/
│  ├─ methods/
│  ├─ transformers/
│  ├─ upscaling/
│  └─ utils/
├─ analysis/                    # Analysis & figure generation
│  ├─ generate_figures/
│  └─ scripts/
├─ dev/
├─ setup.cfg
├─ setup.py
└─ README.md
```

---

## Cite

If you use **forest-age-upscale** or **GAMI**, please cite both the dataset and the paper.

**Dataset (GAMI v2.0):**
```bibtex
@dataset{besnard_gami_v2_2024,
  title     = {Global Age Mapping Integration (GAMI) v2.0: Global Forest Age at 100 m for 2010 and 2020},
  author    = {Besnard, Simon and co-authors},
  year      = {2024},
  doi       = {10.5880/GFZ.1.4.2023.006},
  publisher = {GFZ Data Services}
}
```

**Paper:**
```bibtex
@article{besnard_nee_age_2025,
  title   = {Global covariation of forest age transitions with the net carbon balance},
  author  = {Besnard, S. and Heinrich, V.H.A. and Carvalhais, N. and Ciais, P. and Herold, M. and Luijkx, I. and others},
  journal = {Nature Ecology & Evolution},
  year    = {2025},
  doi     = {10.1038/s41559-025-02821-5}
}
```

---

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch.
3. Make your changes (with clear code, tests, and docs).
4. Open a Pull Request.

---

## License
This project is licensed under the **EUROPEAN UNION PUBLIC LICENCE v.1.2 (EUPL-1.2)**. See [LICENSE](./LICENSE) for details.

---

## Contact
**Simon Besnard** — simon.besnard@gfz.de
