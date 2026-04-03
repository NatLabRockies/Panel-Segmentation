![Panel Segmentation Icon](docs/_static/panel_segmentation_cropped_icon.png)

# Panel-Segmentation

Automated metadata extraction for solar PV installations using satellite imagery and computer vision.

[![PyPI version](https://badge.fury.io/py/panel-segmentation.svg)](https://badge.fury.io/py/panel-segmentation)
[![Build Status](https://github.com/NREL/Panel-Segmentation/actions/workflows/pytest.yaml/badge.svg)](https://github.com/NREL/Panel-Segmentation/actions)
[![Documentation Status](https://readthedocs.org/projects/panel-segmentation/badge/?version=latest)](https://panel-segmentation.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## Overview

Panel-Segmentation provides tools to detect, locate, and characterize solar PV installations from satellite and aerial imagery. It is developed at the [National Renewable Energy Laboratory (NREL)](https://www.nrel.gov/).

**Capabilities include:**

- **Satellite image retrieval** — pull imagery for any lat/lon coordinate via the Google Maps Static API
- **Solar panel detection** — locate panels and retrieve their latitude, longitude, and address within a geographic bounding box (SOL-Searcher Pipeline)
- **Storm damage assessment** — detect and quantify hurricane or hail damage on solar installations from before/after satellite imagery
- **NOAA MESH conversion** — convert NOAA Maximum Estimated Size of Hail (MESH) grib2 files to KML or GeoJSON
- **Tilt and azimuth estimation** — estimate array tilt and azimuth from USGS LiDAR data
- **Mounting configuration classification** — identify roof-mount, ground-mount, carport, and other configurations using an instance segmentation model
- **Site autotagging** — geolocate tracker rows, fixed-tilt rows, and inverters at utility-scale ground-mount sites using a deep learning pipeline

---

## Installation

### Prerequisites

Panel-Segmentation stores deep learning model weights via [Git Large File Storage (LFS)](https://git-lfs.github.com/). Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Linux
sudo apt install git-lfs

# Then initialize
git lfs install
```

### Install from source

```bash
git clone https://github.com/NREL/Panel-Segmentation.git
cd Panel-Segmentation
pip install .
```

### Install MMCV (required for storm damage models)

The storm-related CV models depend on MMCV, which must be built from source:

```bash
pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0
```

> **Note:** MMCV installation may take several minutes.

---

## Documentation

Full documentation — including installation guides, API reference, pipeline tutorials, and example notebooks — is available at **[panel-segmentation.readthedocs.io](https://panel-segmentation.readthedocs.io/en/latest/)**.

---

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/NREL/Panel-Segmentation).

---

## Citation

If you use Panel-Segmentation in your research, please cite it using the information in [`CITATION.cff`](CITATION.cff).

---

## License

Panel-Segmentation is released under the [BSD 3-Clause License](LICENSE).