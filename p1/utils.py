import torch
import pickle

def count_params(model: torch.nn.Module, scenario):
    """
    Count the number of parameters in a given model
    """
    phony = model(*scenario)
    return sum(param.numel() for param in phony.parameters())

def standardize(data):
    """
    Standardize data by substracting mean and dividing by std
    """
    mean, std = data.mean(), data.std()
    return (data - mean)/std

def load_gridsearch(fname):
    """
    Load serialized dict returned from gridsearch
    """
    with open(f"./results/{fname}", "rb") as f:
        return pickle.load(f)
