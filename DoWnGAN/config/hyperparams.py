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
batch_size = 8
gamma = 0.01
content_lambda = 600
#variance_lambda = 8
ncomp = 75
lr = 0.00025

lambdas = { # weights for multi-head critic 
    "joint": 1.0,
    "vars":  [0.25, 0.25, 0.25, 0.25],  # u,v,T,q
    "global": 0.1,                       # or 0.0 initially if you want
}

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
