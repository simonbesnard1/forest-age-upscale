<p align="center">
<a href="https://git.gfz-potsdam.de/besnard/forest_age_upscale">
    <img src="https://media.gfz-potsdam.de/gfz/wv/pic/Bildarchiv/gfz/GFZ-CD_LogoRGB_en.png" alt="Master" height="158px" hspace="10px" vspace="0px" align="right">
  </a>
</p>

***

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
├── data
│   ├── raw            # Raw data
│   ├── processed      # Processed data
├── notebooks          # Jupyter notebooks for experiments and analysis
├── src
│   ├── data           # Data processing scripts
│   ├── models         # Model definitions and training scripts
│   ├── analysis       # Analysis and visualization scripts
├── tests              # Unit tests
├── results            # Results of experiments
├── figures            # Figures for the paper
├── requirements.txt   # Python package dependencies
├── LICENSE
└── README.md
```

## :anger: &nbsp;Package installation
You can install the python package as follows:

***Important: a personal token has to be entered as password for the package installation***
```
pip install git+https://git.gfz-potsdam.de/global-land-monitoring/forest-age-upscale.git

```

## :busts_in_silhouette: &nbsp;Contributing

We welcome contributions to this project. If you would like to contribute, please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -am 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.
- Please make sure your code follows our coding guidelines and passes all tests.

## :email: &nbsp;Contact person
For any questions or inquiries, please contact Simon Besnard (besnard@gfz-potsdam.de)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

