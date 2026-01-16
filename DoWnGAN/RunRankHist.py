#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:10:11 2023

@author: kiridaust
"""

import mlflow
import mlflow.pytorch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from DoWnGAN.GAN.BourgainEmbed import BourgainSampler

device = torch.device("cuda:0")

mod_noise = "/media/data/mlflow_exp/4/fcfc2d47696843f7bdccf16e452727d0/artifacts/Generator/Generator_140"
G = mlflow.pytorch.load_model(mod_noise)
data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_temp/"

cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()

test_data = fine[torch.randint(0,8000,(1,400)),0,...].cpu()
test_data = torch.squeeze(test_data)
z_sampler = BourgainSampler(test_data)

batchsize = 16
invariant = invariant.repeat(batchsize,1,1,1)

random = torch.randint(0, 1000, (40, ))
# mp = torch.nn.MaxPool2d(8)
allrank = []
for sample in random:
    print("Processing",sample)
    coarse_in = coarse[sample,...]
    coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

    gen_out = G(coarse_in, invariant).cpu().detach()
    for i in range(5):
        fine_gen = G(coarse_in, invariant)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    real = fine[sample,0,...].cpu()
    fake = gen_out[:,0,...]
    rankvals = []
    for i in range(128):
        for j in range(128):
            obs = real[i,j].numpy()
            #if(obs != 999):
            ensemble = fake[:,i,j].flatten().numpy()
            allvals = np.append(ensemble,obs)
            rankvals.append(sorted(allvals).index(obs))

    allrank.append(rankvals)
        
l2 = np.array([item for sub in allrank for item in sub])
plt.hist(l2)

plt.imshow(gen_out[36,0,...].cpu())
stdgen = torch.var(gen_out, dim = 0).cpu()
stdgen = torch.squeeze(stdgen)
plt.imshow(stdgen)