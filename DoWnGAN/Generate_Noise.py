# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:51:22 2023

@author: kirid
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
import CRPS as crps
device = torch.device("cuda:0")


### test CRPS metrics
G_toy = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/94c6d5ecb2d84eb085d424cf0c7248e3/artifacts/Generator/Generator_500")
G_real = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/65e9cd4ba68045bdb79526d0196b654e/artifacts/Generator/Generator_500")
data_real = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"
data_toy = "/home/kiridaust/Masters/Data/ToyDataSet/Bimodal_Synth/"

##toy data
coarse_s = np.load(data_toy+"coarse_test.npy")
coarse_s = np.swapaxes(coarse_s, 0, 2)
coarse_s = torch.from_numpy(coarse_s)[:,None,...].to(device).float()
fine_s = np.load(data_toy+"fine_test.npy")
fine_s = np.swapaxes(fine_s, 0, 2)
fine_s = torch.from_numpy(fine_s)[:,None,...].to(device).float()

batchsize = 16
coarse_s_samp = coarse_s[42,...].unsqueeze(0).repeat(batchsize,1,1,1)
Gen_S = G_toy(coarse_s_samp).cpu().detach()

##real wind
cond_fields = xr.open_dataset(data_real + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_real + "fine_test.nc", engine="netcdf4")
coarse_r = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine_r = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_real + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()

coarse_r_samp = coarse_r[42,...].unsqueeze(0).repeat(batchsize,1,1,1)
invariant = invariant.repeat(batchsize,1,1,1)
Gen_R = G_real(coarse_r_samp,invariant).cpu().detach()

GR = Gen_R[:,0,...]
GS = Gen_S[:,0,...]
RR = fine_r[42,0,...]
RS = fine_s[42,0,...]

crps.CRPS(GS[:,11,110].flatten(), RS[11,110].cpu()).compute()
crps.CRPS(GR[:,11,110].flatten(), RR[11,110].cpu()).compute()
#################################3
##generate temperature fields
G = mlflow.pytorch.load_model("Generators/dbba1469156c44ae8ebeaa4a239ecef9/artifacts/Generator/Generator_500")
data_folder = "../Data/ds_temphumid/"

cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()
batchsize = 32
invariant = invariant.repeat(batchsize,1,1,1)

fine_sm = fine[0:5,0,...]
test = fine_sm[0,...]-fine_sm[1,...]
t1 = torch.argsort(fine_sm, dim = 0)
plt.imshow(t1[0,...].cpu())
plt.show()
test_hist = torch.histc(t1[0,...],bins = 5)
test_hist2 = torch.histc(t1[1,...],bins = 5)
t2 = torch.add(test_hist,test_hist2)

allgen = []
for i in range(32):
    print(".", end = " ")
    gen = G(coarse[(i*32):((i+1)*32),...],invariant).cpu().detach()
    allgen.append(gen)
    del gen
    
gen2 = torch.cat(allgen,0)
torch.save(gen2, "JustTemp_Gens_First1024Validation.pt")

#####################################################################################
### Compare CRPS Metrix of wind with regular and variance loss
# G_var = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/b89f2ea32c7d49e1bcc3595e37590fc9/artifacts/Generator/Generator_80")
# G_reg = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/7cf2b23e9bf44b4d92138d0e4c1e2486/artifacts/Generator/Generator_80")
# data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"
# 
# cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
# fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
# coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
# invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()
# 
# batchsize = 32
# invariant = invariant.repeat(batchsize,1,1,1)
# 
# random = torch.randint(0, 1000, (30, ))
# # mp = torch.nn.MaxPool2d(8)
# allrank = []
# for sample in random:
#     print("Processing",sample)
#     coarse_in = coarse[sample,...]
#     coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)
# 
#     gen_out = G_var(coarse_in, invariant).cpu().detach()
#     for i in range(5):
#         fine_gen = G_var(coarse_in, invariant)
#         gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
#         del fine_gen
#     
#     real = fine[sample,0,...].cpu()
#     fake = gen_out[:,0,...]
#     # real = mp(real.unsqueeze(0))
#     # fake = mp(gen_out[:,0,...])
#     rankvals = []
#     for i in range(128):
#         for j in range(128):
#             obs = real[i,j].numpy()
#             ensemble = fake[:,i,j].flatten().numpy()
#             rankvals.append(crps.CRPS(ensemble,obs).compute()[0])
#             
#     allrank.append(rankvals)
# 
# var_crps = np.array([item for sub in allrank for item in sub])
# 
# plt.boxplot(reg_crps)
# plt.boxplot(var_crps)
# dat = np.vstack([reg_crps, var_crps])
# np.save("CRPS_Comp.npy",dat)

########################################
#mod_noise = "/media/data/mlflow_exp/4/94c6d5ecb2d84eb085d424cf0c7248e3/artifacts/Generator/Generator_500"

#mod_noise = "/media/data/mlflow_exp/4/6f3f3b042510493cb385939d57864de3/artifacts/Generator/Generator_480" ##dem+10
mod_noise = "/media/data/mlflow_exp/4/b190fb9c6b63458e9152c6b7706cb1f8/artifacts/Generator/Generator_250"
mod_noise = "/media/data/mlflow_exp/4/971937cd44424b438c113ead26c4384d/artifacts/Generator/Generator_500"
G = mlflow.pytorch.load_model(mod_noise)
data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_temp/"


# coarse = np.load(data_folder+"coarse_val.npy")
# coarse = np.swapaxes(coarse, 0, 2)
# coarse = torch.from_numpy(coarse)[:,None,...].to(device).float()
# fine = np.load(data_folder+"fine_val.npy")
# fine = np.swapaxes(fine, 0, 2)
# fine = torch.from_numpy(fine)[:,None,...].to(device).float()
# 
# plt.imshow(coarse[4,0,...].cpu())
# plt.show()
# invar = np.load(data_folder+"dem_crop.npy")
# invariant = torch.from_numpy(invar)[None,...].to(device).float()

cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()


batchsize = 16
sample = 5
invariant = invariant.repeat(batchsize,1,1,1)

coarse_in = coarse[0:batchsize,...].to(device).float()
gen = G(coarse_in, invariant)
gen.shape
del gen
gen = gen.cpu().detach()
plt.imshow(gen[0,0,...])
plt.show()
plt.close()
plt.imshow(gen[3,1,...])
plt.show()
plt.imshow(fine[3,1,...])
plt.show()

coarse_in = coarse[sample,...]
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

gen_out = G(coarse_in, invariant)
real = fine[sample,...]
from DoWnGAN.GAN.losses import crps_empirical

res = crps_empirical(gen_out, real)
torch.mean(res)

gentest = gen_out[0,...]
real = torch.squeeze(real)
plt.imshow(gentest.cpu().detach())

gen_out[gen_out > torch.unsqueeze(real, 0)] = -999

plt.imshow(gen_out[3,...].cpu().detach())

for i in range(5):
    fine_gen = G(coarse_in, invariant)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

plt.imshow(gen_out[53,0,...])
plt.imshow(gen_out[1,0,...])
plt.imshow(gen_out[124,0,...])
plt.imshow(gen_out[87,0,...])
stoch_var = torch.var(gen_out, dim = 0)
plt.imshow(stoch_var[0,...])
plt.imshow(invariant[0,0,...].cpu())
plt.imshow(fine[sample,0,...].cpu())
plt.imshow(coarse_in[0,0,...].cpu())
# noise_f = torch.normal(0,1,size = [batchsize,1,128,128], device=device)
# invariant = torch.cat([invariant, noise_f], 1)
# print(invariant.size())

random = torch.randint(0, 1000, (40, ))
# mp = torch.nn.MaxPool2d(8)
allrank = []
for sample in random:
    print("Processing",sample)
    coarse_in = coarse[sample,...]
    coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1).to(device).float()

    gen_out = G(coarse_in, invariant).cpu().detach()
    for i in range(24):
        fine_gen = G(coarse_in, invariant)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    real = fine[sample,0,...]
    #lowerq = torch.quantile(real, 0.1)
    #real[real < torch.quantile(real, 0.5)] = 999 ##just test for high values areas
    #real[real < lowerq] = 999
    fake = gen_out[:,0,...]
    # real = mp(real.unsqueeze(0))
    # fake = mp(gen_out[:,0,...])
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
#np.save("Rank_Hist_Data_Synth_DEM_w10.npy", l2)
plt.hist(l2)
plt.show()
merid = l2

###plot marginal distributions

batchsize = 16
coarse_in = torch.mean(coarse,0)
plt.imshow(coarse_in[0,...].cpu())
plt.show()
for i in range(4):
  plt.close()
  plt.imshow(fine[i,0,...].cpu())
  plt.show()
  
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

gen_out = G(coarse_in).cpu().detach()
for i in range(4):
  plt.close()
  plt.imshow(gen_out[i,0,...])
  plt.show()
  
for i in range(10):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen


xp = 5
yp = 5
samp1 = gen_out[:,0,xp,yp].flatten()
samp2 = fine[0:192,0,xp,yp].cpu().flatten()
plt.close()
sns.kdeplot(samp1,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

torch.quantile(fine[0:250,...].cpu(), torch.tensor([0.01,0.1,0.5,0.9,0.99]))

from scipy.stats import wasserstein_distance
coarse_in = coarse[42,...]
coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)
gen_out = G(coarse_in, invariant).cpu().detach()
for i in range(16):
    fine_gen = G(coarse_in, invariant)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
wass_reg = np.zeros([128,128])
for i in range(128):
    for j in range(128):
        real = fine[0:512,0,i,j].cpu().flatten().numpy()
        gen = gen_out[:,0,i,j].flatten().numpy()
        wass_reg[i,j] = wasserstein_distance(gen,real)
        
wass_reg[wass_reg > np.quantile(wass_reg,0.95)] = np.quantile(wass_reg,0.95)
plt.imshow(wass_reg,cmap="viridis")
        
wass_out = np.zeros([128,128])
for i in range(128):
    for j in range(128):
        real = fine[0:512,0,i,j].cpu().flatten().numpy()
        gen = gen_out[:,0,i,j].flatten().numpy()
        real_std = (real - np.mean(real))/np.var(real)
        gen_std = (gen - np.mean(gen))/np.var(gen)
        wass_out[i,j] = wasserstein_distance(gen_std,real_std)
        
wass_out[wass_out > np.quantile(wass_out,0.95)] = np.quantile(wass_out,0.95)
plt.imshow(wass_out,cmap="viridis")
        
################Compare PacGAN#######################################
mod_1 = "/media/data/mlflow_exp/4/97dcb6fdf104430b8243f7bf9c46326c/artifacts/Generator/Generator_500"
mod_2 = "/media/data/mlflow_exp/4/65e9cd4ba68045bdb79526d0196b654e/artifacts/Generator/Generator_500"
G1 = mlflow.pytorch.load_model(mod_1)
G2 = mlflow.pytorch.load_model(mod_2)

realf = fine[0:32,0,...].flatten()
gen_out = G1(coarse[0:32,...], invariant).cpu().detach()
for i in range(1,32):
    fine_gen = G1(coarse[(i*32):((i+1)*32),...], invariant)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
np.save("PacGAN_GEN.npy",gen_out)
fake_pac = gen_out

del gen_out
gen_out = G2(coarse[0:32,...], invariant).cpu().detach()
for i in range(1,32):
    fine_gen = G2(coarse[(i*32):((i+1)*32),...], invariant)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
np.save("RegularGAN_GEN.npy",gen_out)
np.save("WRF_Wind.npy",fine[0:1024,...].cpu())
fake_reg = gen_out

torch.var(fake_reg)
torch.var(fake_pac)
torch.var(fine[0:192,...])

torch.var(fake_reg[:,1,...])
torch.var(fake_pac[:,1,...])
torch.var(fine[0:192,1,...])

torch.var(fake_reg[:,0,...])
torch.var(fake_pac[:,0,...])
torch.var(fine[0:192,0,...])
# #torch.save(gen_out,"Wind_NoiseInject_6884.pt")

# for i in range(5):
#     plt.imshow(gen_out[i,0,...])
#     plt.show()
    

# gen_mean = torch.mean(gen_out,0)
# plt.imshow(gen_mean[1,...])
# plt.imshow(coarse_in[0,...].cpu())
# plt.imshow(fine[sample,0,...].cpu())
# print(coarse_in.size())

# torch.manual_seed(0)
# random = torch.randint(0, 32, (20, ))
# plt.imshow(coarse[9,0,...].cpu())
# fake = G(coarse[0:32,...],invariant)
# sampnum = 31
# plt.imshow(coarse[sampnum,0,...].cpu())
# plt.imshow(fake[sampnum,0,...].cpu().detach())
# plt.imshow(fine[sampnum,0,...].cpu().detach())
# gen_cov = fine_gen
# gen_inject = fine_gen

# plt.rcParams["figure.figsize"] = (7,6)
# plt.imshow(fine_gen[1,0,...])

# plt.savefig('Example_PFS_NoiseInject.svg',bbox_inches = 'tight')
# #plt.imshow(fine_gen[2,0,...])
# #plt.imshow(fine_gen[3,0,...])
# #plt.imshow(fine_gen[42,0,...])


# plt.savefig('RankHist_PFS_NoiseInject.svg', bbox_inches = 'tight', dpi = 600)
# sdres = torch.std(fine_gen,dim = 0)
# plt.imshow(sdres[0,...])

# #torch.save(fine_gen,"Wind_Generated_d4c12d8ef6b84871bc0cb5fd18d638ef.pt")
# #plt.imshow(fine_gen[1,0,...])
# #plt.imshow(fine_gen[56,0,...])
# # plt.imshow(fine_gen[233,0,...])



# sample = fine_gen[42,0,...].flatten()
# sns.set_style('whitegrid')
# sns.kdeplot(sample,bw = 0.5)
#real = fine_in[:,:,xp,yp].flatten()
#myplt = sns.kdeplot(real,bw = 0.5)
#myplt.figure.savefig("ToyDat2_Figure.png")
print("Done")

# class NetCDFSR(Dataset):
#     """Data loader from torch.Tensors"""
#     def __init__(
#         self,
#         coarse: torch.Tensor,
#         fine: torch.Tensor,
#         invarient: torch.Tensor,
#         device: torch.device,
#     ) -> torch.Tensor:
#         """
#         Initializes the dataset.
#         Returns:
#             torch.Tensor: The dataset batches.
#         """
#         self.fine = fine
#         self.coarse = coarse
#         self.invarient = invarient

#     def __len__(self):
#         return self.fine.size(0)

#     def __getitem__(self, idx):

#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         fine_ = self.fine[idx,...]
#         coarse_ = self.coarse[idx,...]
#         invarient_ = self.invarient
#         return coarse_, fine_, invarient_

# mod_noise = "/media/data/mlflow_exp/4/b56771fd635d414d9586dc72019237be/artifacts/Generator/Generator_420"
# G = mlflow.pytorch.load_model(mod_noise)

# cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
# fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
# coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
# invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

# ds = NetCDFSR(coarse, fine, invariant, device=device)
# dataloader = torch.utils.data.DataLoader(
#     dataset=ds, batch_size=16, shuffle=False
# )

# data_in = None
# for i in range(225):
#     data_in = next(iter(dataloader))

# temp = G(data_in[0],data_in[2])
# noise_gen = temp[0,...].cpu().detach()
# for i in range(100):
#     torch.cuda.empty_cache()
#     print("Generating",i)
#     temp = G(data_in[0],data_in[2])
#     noise_gen = torch.cat((noise_gen,temp[0,...].cpu().detach()),0)
#     del temp

# print(noise_gen.size())
# torch.save(noise_gen,"ExampleNoiseGen.pt")
# print("Done")


# ############
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gen = torch.load("Wind_NoiseInject_6884.pt")
b1 = gen[0:32,...]
b2 = torch.reshape(b1,(8,8,128,128))
plt.imshow(b1[1,0,...])
plt.imshow(b2[0,2,...])

plt.imshow(b1[1,1,...])

plt.imshow(b2[7,7,...])
plt.imshow(b1[31,1,...])
#t1 = gen[0:10,...]
#t2 = gen[10:20,...]
#v1 = torch.var(t1,(2,3))
#v2 = torch.var(t2,(2,3))
#var_loss = torch.mean(torch.abs(v1 - v2))

#plt.imshow(gen[0,0,...])
#plt.imshow(gen[42,0,...])
#plt.imshow(gen[175,0,...])
#plt.imshow(gen[130,0,...])

#avg = torch.mean(gen,0)

# fig, ax = plt.subplots(1, 3)

# for i,num in enumerate([0,42,175]):
#     ax[i].imshow(gen[num,0,...])
#     ax[i].axis('off')
# fig.show()
# np.save("GAN_Noise_Res.npy",gen)

# tfine = np.load("C:/Users/kirid/Desktop/Masters/ToyDataSet/fine_train.npy")
# t1 = np.swapaxes(tfine, 0, 2)
# plt.imshow(t1[0,...])

