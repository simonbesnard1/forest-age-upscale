# Python package for model training and upscaling forest age product
<p align="center">
<a href="https://git.gfz-potsdam.de/besnard/forest_age_upscale">
    <img src="https://media.gfz-potsdam.de/gfz/wv/pic/Bildarchiv/gfz/GFZ-CD_LogoRGB_en.png" alt="Master" height="158px" hspace="10px" vspace="0px" align="right">
  </a>
</p>

***

## :memo: &nbsp;What is it?
This Github repo contains all model training and inference modules for upscaling of forest age products

## :anger: &nbsp;Package installation
You can install the python package as follows:
```
pip install git+https://git.gfz-potsdam.de/{USERNAME}/forest_age_upscale.git
```

## :notebook_with_decorative_cover: &nbsp;Getting started

#### 1. Perform a cross-validation study

```
#%% Load package
from ageUpscaling.core.study import Study
```

```
#%% Define study
DataConfig_path= "./experiments/data_config.yaml"
study_ = Study(DataConfig_path = DataConfig_path,
                  exp_name  = 'MLPregressor',
                  out_dir= './output/',
                  n_jobs = 10)
```
 
```
#%% Run cross-validation study
study_.cross_validation(n_folds=10, 
			valid_fraction=0.3, 
			feature_selection=False,
                       feature_selection_method= 'recursive')
```

## :busts_in_silhouette: &nbsp;Contributing
If you find something which doesn't make sense, or something doesn't seem right, please make a pull request and please add valid and well-reasoned explanations about your changes or comments.

Before adding a pull request, please see the **[contributing guidelines](.github/CONTRIBUTING.md)**. You should also remember about this:

All **suggestions/PR** are welcome!

### Code Contributors
This project exists thanks to all the people who contribute.

## :email: &nbsp;Contact person
Simon Besnard (besnard@gfz-potsdam.de)

