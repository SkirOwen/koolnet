# Koopman Operator Object Locator Net - (KOOLNet)

![Static Badge](https://img.shields.io/badge/python-blue?logo=python&logoColor=yellow)
![GitHub](https://img.shields.io/github/license/SkirOwen/pycylinder_flow?color=green)
![Static Badge](https://img.shields.io/badge/pytorch-grey?logo=pytorch&logoColor=red)


This project is to detect the location of an obstacle responsible for a
turbulent flow in the context of tidal turbine.

The code expected as input the koopman mode decomposition of a flow, the following
code was used:
https://github.com/SkirOwen/ResDMDpy

## Installation

The code was created using `python=3.11`, it should support `3.8` but has not been tested, nor it is guaranteed
it will stay compatible.
We recommend using `conda` or [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) to install
the dependencies.  
You can install with `pip`. However, since we are dependent on `rasterio`, that requires C compiled code,
we cannot guarantee the installation with `pip` on Windows. 

### CONDA | CUDA
```shell
conda env create -f environment_gpu.yml
```

### CONDA | no CUDA
```shell
conda env create -f environment_no_gpu.yml
```

### PIP
You may want to install it in a virtual environment.
```shell
pip install -r requirement.txt
```
