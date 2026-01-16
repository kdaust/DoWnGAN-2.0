# Calculates epoch losses and logs them
from DoWnGAN.GAN.losses import content_loss, content_MSELoss, SSIM_Loss, rankhist_loss
from DoWnGAN.config import config
import DoWnGAN.config.hyperparams as hp
import DoWnGAN.GAN.stage as s

import csv
import mlflow
from mlflow import log_param, log_metric

import torch
import os
import pandas as pd

from csv import DictWriter

mlflow.set_tracking_uri(config.EXPERIMENT_PATH)

def log_to_file(dict, train_test):
    """Writes the metrics to a csv file"""
    csv_path = f"{mlflow.get_artifact_uri()}/{train_test}_metrics.csv"
    # This will write to a new csv file if there isn't one
    # but append to an existing one if there is one
    with open(csv_path, "a", newline="") as f:
        df = pd.DataFrame.from_dict(data=dict)
        df.to_csv(f, header=(f.tell()==0))
    # mlflow.log_artifact(csv_path)


def initialize_metric_dicts(d, num_preds):
    for key in hp.metrics_to_calculate.keys():
        if(key == "MAE"):
            for i in range(num_preds):
                d[key + "_" + str(i)] = []
        else:
            d[key] = []
    return d


def metric_print(metric, metric_value):
    print(f"{metric}: {metric_value}")


def post_epoch_metric_mean(d, train_test):
    # Tracks batch metrics through 
    means = {}
    for key in d.keys():
        means[key] = [torch.mean(
            torch.FloatTensor(d[key])
        ).item()]
        log_metric(f"{key}_{train_test}", means[key][0])
        metric_print(f"{key}_{train_test}", means[key][0])

    log_to_file(means, train_test)


def gen_batch_and_log_metrics(G, C, coarse, real, invariant, d):

    def critic_score(s_joint, s_vars, s_global):
        D = hp.lambdas["joint"] * s_joint.mean(dim=(1, 2, 3))  # (B,)
        for i, s in enumerate(s_vars):
            D = D + hp.lambdas["vars"][i] * s.mean(dim=(1, 2, 3))
        if s_global is not None:
            D = D + hp.lambdas.get("global", 0.0) * s_global.view(-1)
        return D

    fake = G(coarse,invariant).detach()
    s_joint_real, s_vars_real, s_global_real = C(real,invariant,coarse)
    s_joint_fake, s_vars_fake, s_global_fake = C(fake,invariant,coarse)

    creal = critic_score(s_joint_real, s_vars_real, s_global_real)
    cfake = critic_score(s_joint_fake, s_vars_fake, s_global_fake)
    for key in hp.metrics_to_calculate.keys():
        if key == "Wass":
            d[key].append(hp.metrics_to_calculate[key](creal.mean().detach(), cfake.mean().detach(), config.device).detach().cpu().item())
        elif key == "CRPS":
            d[key].append(hp.metrics_to_calculate[key](G,coarse, real, invariant, config.device).cpu().item())
        else:
            for i in range(real.shape[1]):
                d[key + "_" + str(i)].append(hp.metrics_to_calculate[key](real[:,i,...], fake[:,i,...], config.device).detach().cpu().item())
    return d

def log_network_models(C, G, epoch):
    #mlflow.pytorch.log_model(C, f"Critic/Critic_{epoch}")
    #mlflow.pytorch.log_state_dict(C.state_dict(), f"Critic/Critic_{epoch}")
    g_path = f"/users/kdaust/DoWnGAN_Kiri/Generators/Generator_{epoch}.pt"
    G.save(g_path)