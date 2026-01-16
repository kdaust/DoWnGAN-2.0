# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:30:20 2023

@author: kirid
"""

import mlflow
import mlflow.pytorch
import torch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

import scipy.stats

class NetCDFSR(Dataset):
    """Data loader from torch.Tensors"""
    def __init__(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
        invarient: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initializes the dataset.
        Returns:
            torch.Tensor: The dataset batches.
        """
        self.fine = fine
        self.coarse = coarse
        self.invarient = invarient

    def __len__(self):
        return self.fine.size(0)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        invarient_ = self.invarient
        return coarse_, fine_, invarient_

G = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/c3367f222c2a4e24882cd3fd3f8bc52e/artifacts/Generator/Generator_200")
#G = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/17d56bf78c714b18a44ae2f5116d0d15/artifacts/Generator/Generator_410")


cond_fields = xr.open_dataset("~/Masters/Data/temperature/coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/temperature/fine_test.nc", engine="netcdf4")
invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
#invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Coarse.nc", engine = "netcdf4")

coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

ds = NetCDFSR(coarse, fine, invariant, device=device)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=16, shuffle=True
)

torch.cuda.empty_cache()

for i, data in enumerate(dataloader):
    if(i >= 1):
        break
    print("running batch ", i)
    #torch.cuda.empty_cache()
    out = G(data[0],data[2]).cpu().detach()
    torch.save(out, "generated.pt")
    real = data[1].cpu().detach()
    torch.save(real, 'real.pt')

    #print("RALSD: ",log_dist)
    del data
    #del out
    #del rea
    
import matplotlib.pyplot as plt
temp_gen = out[14,1,...]
temp_real = real[14,1,...]
plt.imshow(temp_gen)
plt.imshow(temp_real)

