# Defines the hyperparameter and constants configurations
from DoWnGAN.GAN.losses import (
    content_loss,
    content_MSELoss,
    SSIM_Loss,
    wass_loss,
    crps_loss
)

import torch.nn as nn
import torch

import os


# Hyper params
gp_lambda = 10
critic_iterations = 5
batch_size = 6
gamma = 0.01
content_lambda = 400
drift_epsilon = 1e-4
#variance_lambda = 8
ncomp = 75
lr = 0.00025

lambdas = {
    "joint": 1.0,
    "vars":  [0.20, 0.20, 0.20, 0.20, 0.20],  # u,v,T,q,p
    "global": 0.1,
}

#crps_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=torch.device("cuda:0"))
# Run configuration parameters
epochs = 251
print_every = 10
save_every = 100
use_cuda = True

# Frequency separation parameters
filter_size = 7
padding = filter_size // 2
low = nn.AvgPool2d(filter_size, stride=1, padding=0)
rf = nn.ReplicationPad2d(padding)


metrics_to_calculate = {
    "MAE": content_loss,
    #"MSE": content_MSELoss,
    #"VLoss": variance_loss,
    "Wass": wass_loss
}
