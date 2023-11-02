import os
import re
import setuptools

PKG_NAME = "ageUpscaling"

HERE = os.path.abspath(os.path.dirname(__file__))

PATTERN = r'^{target}\s*=\s*([\'"])(.+)\1$'

AUTHOR = re.compile(PATTERN.format(target='__author__'), re.M)
VERSION = re.compile(PATTERN.format(target='__version__'), re.M)
LICENSE = re.compile(PATTERN.format(target='__license__'), re.M)
AUTHOR_EMAIL = re.compile(PATTERN.format(target='__author_email__'), re.M)


def parse_init():
    with open(os.path.join(HERE, PKG_NAME, '__init__.py')) as f:
        file_data = f.read()
        print(file_data)
    return [regex.search(file_data).group(2) for regex in
            (AUTHOR, VERSION, LICENSE, AUTHOR_EMAIL)]


with open("README.md", "r") as fh:
    long_description = fh.read()

author, version, license, author_email = parse_init()

setuptools.setup(
    name=PKG_NAME,
    author=author,
    author_email=author_email,
    license=license,
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.gfz-potsdam.de/global-land-monitoring/forest-age-upscale",
    packages=setuptools.find_packages(include=['ageUpscaling',
                                               'ageUpscaling.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.5',
    install_requires=[
        "xarray==2023.10.1",
        "zarr==2.10.3",
        "rioxarray==0.15.0",
        "netCDF4==1.6.5",
        "dask==2023.10.0",
        "scikit-learn==1.3.2",
        "Boruta==0.3",
        "numpy==1.26",
        "optuna==3.4.0",
        "PyYAML==6.0.1",
        "matplotlib==3.8.0",
        "bottleneck==1.3.7",
        "rasterio==1.3.9",
        "xgboost==1.7.2",
        "torch==2.1.0",
        "TPOT==0.12.1",
        "shap==0.43.0",
        'tqdm==4.66.1',
        "dask_jobqueue==0.8.2",
        #"autogluon==0.8.2",
        "cartopy==0.22.0",
        "shapely==2.0.2",
        "geopandas==0.14.0",
        "scipy==1.11.3",
        "sklearn-genetic-opt==0.10.1"],
    include_package_data=True,
)
