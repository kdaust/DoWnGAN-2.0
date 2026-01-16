from mlflow import log_param
import DoWnGAN.config.hyperparams as hp
import logging

def log_hyperparams():
    """Logs the hyperparameters"""
    keys = [item for item in dir(hp) if not item.startswith("__")]
    values = hp.__dict__
    for key in keys:
        log_param(key, values[key])


def define_experiment(mlclient):
    #[print(exp.experiment_id,":", exp.name) for i, exp in enumerate(mlclient.list_experiments())]
    set_exp = "Multivar"
    #mlclient.create_experiment(set_exp)
    return mlclient.get_experiment_by_name(set_exp).experiment_id

def write_tags():
    choice = "(16x16) just Precip, LR topography"
    return choice
