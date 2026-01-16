
import mlflow
import mlflow.pytorch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import pandas as pd
device = torch.device("cuda:0")


#mod_noise = "/media/data/mlflow_exp/4/9625d9c4e7584218827d4ec1740eb7f0/artifacts/Generator/Generator_500"
#G = mlflow.pytorch.load_model(mod_noise)

data_folder = "/home/kiridaust/Masters/Data/ToyDataSet/Bimodal_Synth/"

# cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
# coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
# invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()
coarse = np.load(data_folder+"coarse_val.npy")
coarse = np.swapaxes(coarse, 0, 2)
coarse = torch.from_numpy(coarse)[:,None,...].to(device).float()
fine = np.load(data_folder+"fine_val.npy")
fine = np.swapaxes(fine, 0, 2)
fine = torch.from_numpy(fine)[:,None,...]

batchsize = 32
#invariant = invariant.repeat(batchsize,1,1,1)
# noise_f = torch.normal(0,1,size = [batchsize,1,128,128], device=device)
# invariant = torch.cat([invariant, noise_f], 1)
# print(invariant.size())

coarse_in = torch.mean(coarse,0)
print(coarse_in.size())
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

models = ['6bb8521944654654bf78a576398b4f80/artifacts/Generator/Generator_450','6bb8521944654654bf78a576398b4f80/artifacts/Generator/Generator_500']
modNm = ['Epoch450', 'Epoch500']
#modDat = [coarse_noise, coarse_in]
##wasserstien distance

xp = 5
yp = 5

res = dict()
for i in range(len(modNm)):
    print("Analysing model",modNm[i])
    mod_noise = "/media/data/mlflow_exp/4/" + models[i]
    G = mlflow.pytorch.load_model(mod_noise)
    # if(i == -1):
    #     noise_c = torch.normal(0,1,size = [batchsize, 1, coarse_in.shape[2],coarse_in.shape[3]], device=device)
    #     curr_coarse = torch.cat([coarse_in, noise_c], 1)
    # else:
    curr_coarse = coarse_in
    gen_out = G(curr_coarse).cpu().detach()
    for j in range(32):
        fine_gen = G(curr_coarse)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    dat1 = gen_out[:,0,xp,yp].flatten()
    res[modNm[i]] = dat1
    del gen_out
    del G
    del curr_coarse
    
for nm in modNm:
    sns.kdeplot(res[nm], label = nm)
sns.kdeplot(fine[:,0,xp,yp].flatten(),label = "Real")
plt.legend()
#plt.show()
plt.xlabel("Value")
plt.savefig("Marginal_CompareEpoch_Toydat_Bimodal.svg", dpi = 600)
print("done!")

res = dict()
for i in range(len(modNm)):
    currdat = []
    print("Analysing model",modNm[i])
    mod_noise = "/media/data/mlflow_exp/4/" + models[i] +"/artifacts/Generator/Generator_500"
    G = mlflow.pytorch.load_model(mod_noise)
    if(i == -1):
        noise_c = torch.normal(0,1,size = [batchsize, 1, coarse_in.shape[2],coarse_in.shape[3]], device=device)
        curr_coarse = torch.cat([coarse_in, noise_c], 1)
    else:
        curr_coarse = coarse_in
    gen_out = G(curr_coarse).cpu().detach()
    for j in range(16):
        fine_gen = G(curr_coarse)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    for y in range(128):
        for x in range(128):
            d1 = gen_out[:,0,x,y].flatten()
            d2 = fine[:,0,x,y].flatten()
            currdat.append(wasserstein_distance(d1, d2))
    
    res[modNm[i]] = currdat
    del gen_out
    del G
    del curr_coarse

df = pd.DataFrame(res)
df2 = pd.melt(df,value_vars=modNm)
sns.violinplot(data = df2, x = 'variable', y = 'value',scale='count')
plt.xlabel("Model")
plt.ylabel("Wasserstein Distance")