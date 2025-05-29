# Installs
## Create and activate conda environment
```
conda create -n aurora python=3.10
conda activate aurora
``` 

## Install packages (ensure access to GPU when installing pytorch)
```
conda install conda-forge::numpy -y
conda install pytorch::pytorch -y
conda install conda-forge::huggingface_hub -y
conda install conda-forge::einops -y
conda install conda-forge::timm -y
conda install anaconda::xarray -y
pip install zarr
conda install anaconda::scipy
conda install conda-forge::timm -y
conda install conda-forge::hydra-core -y
```

## Check pytorch install
```
import torch 
torch.cuda.is_available()
```