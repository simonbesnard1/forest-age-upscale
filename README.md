<p align="center">
<a href="https://git.gfz-potsdam.de/besnard/forest_age_upscale">
    <img src="https://media.gfz-potsdam.de/gfz/wv/pic/Bildarchiv/gfz/GFZ-CD_LogoRGB_en.png" alt="Master" height="158px" hspace="10px" vspace="0px" align="right">
  </a>
</p>

***
# Global covariation of forest age transitions with the net carbon balance #

## :notebook_with_decorative_cover: &nbsp;Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## :memo: &nbsp;Overview
This repository contains the code and data associated with the scientific paper titled "Global covariation of forest age transitions with the net carbon balance". The paper investigates changes in forest age distribution from 2010 to 2020 and their implications for the carbon cycle.

## Repository Structure

```plaintext
├── ageUpscaling 		# Core of the package
│   ├── core
│   ├── cubegen
│   ├── dataloaders
│   ├── diagnostic
│   ├── fidc_cube
│   ├── methods
│   ├── transformers
│   ├── upscaling
│   └── utils
├── analysis			# Scripts to run all the analyis and generate the figures
│   ├── generate_figures
│   └── scripts
├── dev
├── README.md
├── setup.cfg
└── setup.py
```

## :anger: &nbsp;Package installation

The code requires `python>=3.11`

Install forest-age-upscale:

```
pip install git+https://git.gfz-potsdam.de/global-land-monitoring/forest-age-upscale.git

```

or clone the repository locally and install with

```
git clone git@git.gfz-potsdam.de:global-land-monitoring/forest-age-upscale.git
cd forest-age-upscale; pip install -e .
```

## :busts_in_silhouette: &nbsp;Contributing

We welcome contributions to this project. If you would like to contribute, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## :email: &nbsp;Contact person
For any questions or inquiries, please contact Simon Besnard (besnard@gfz-potsdam.de)

## License
This project is licensed under the EUROPEAN UNION PUBLIC LICENCE v.1.2 License - see the LICENSE file for details.

## Citing forest-age-upscale

If you use forest-age-upscale in your research, please use the following BibTeX entry.

```
@article{besnard2024,
  title={Forest Age Upscale},
  author={Besnard, Simon},
  doi={},
  year={2024}
}
```

