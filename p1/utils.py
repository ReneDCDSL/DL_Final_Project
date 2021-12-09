import torch
import numpy as np

def count_params(model: torch.nn.Module, scenario):
    """
    Count the number of parameters in a given model
    """
    phony = model(*scenario)
    return sum(param.numel() for param in phony.parameters())

def get_idx_best_acc(df):
    """
    Get the index of the best accuracy wrt to the standard deviation
    The fewer the number of parameters the better

    Made into a function, so we can get the best id by not just taking the highest accuracy (and taking into account the std or the nb of params for example)
    """
    # return np.argmin(df["nb_params"][df["accuracy"] <= df.loc[index_best, "accuracy"] + df.loc[index_best, "std"]])
    return np.argmax(df["accuracy"])

def standardize(data):
    """
    Standardize data by substracting mean and dividing by std
    """
    mean, std = data.mean(), data.std()
    return (data - mean)/std
