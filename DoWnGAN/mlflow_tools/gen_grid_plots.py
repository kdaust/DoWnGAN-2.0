# Plots matplotlib grids and saves to file
import torch
import matplotlib.pyplot as plt
import torchvision
import DoWnGAN.config.hyperparams as hp
from DoWnGAN.config import config
import mlflow

def gen_grid_images(G, coarse, invariant, real, epoch, train_test):
    """
    Plots a grid of images and saves them to file
    Args:
        coarse (torch.Tensor): The coarse input.
        fake (torch.Tensor): The fake input.
        real (torch.Tensor): The real input.
    """
    random = torch.randint(0, hp.batch_size, (5, ))
    # if(invariant == -1):
    #     fake = G(coarse[random, ...].to(config.device))
    # else:
    fake = G(coarse[random, ...].to(config.device), invariant[random,...].to(config.device))
    
    coarse = torchvision.utils.make_grid(
        coarse[random, ...],
        nrow=5
    )[4, ...]

    fake = torchvision.utils.make_grid(
        fake,
        nrow=5
    )[3, ...]


    real = torchvision.utils.make_grid(
        real[random, ...],
        nrow=5
    )[3, ...]


    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Training Samples")

    # Plot the coarse and fake samples
    subfigs = fig.subfigures(nrows=3, ncols=1)
    
    # Coarse Samples
    subfigs[0].suptitle("Coarse ERA5")
    ax = subfigs[0].subplots(1, 1)
    ax.imshow(coarse.cpu().detach(), origin="lower")

    # Generated fake
    subfigs[1].suptitle("Generated")
    ax = subfigs[1].subplots(1, 1)
    ax.imshow(fake.cpu().detach(), origin="lower")

    # Ground Truth
    subfigs[2].suptitle("WRF")
    ax = subfigs[2].subplots(1, 1)
    ax.imshow(real.cpu().detach(), origin="lower")


    if epoch % 5 == 0:
        plt.savefig(f"{mlflow.get_artifact_uri()}/{train_test}_{epoch}.png")
    plt.savefig(f"{mlflow.get_artifact_uri()}/{train_test}.png")
    plt.close(fig)
