#%%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlc_practical_prologue import generate_pair_sets
from models import Siamese, Baseline

def count_params(scenario):
    """
    Count the number of parameters in a given model
    """
    phony = Siamese(scenario[0], scenario[1], scenario[2])
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

def siamese_accuracy(model, data_input, data_target):
    """
    Compute accuracy of siamese network
    """
    
    # Generate predictions
    with torch.no_grad():
        pred2, (pred10_1, pred10_2) = model(data_input)

    # Get best predictions
    pred_class2 = torch.argmax(pred2, 1)
    pred_class10_1 = torch.argmax(pred10_1, 1)
    pred_class10_2 = torch.argmax(pred10_2, 1)
    pred_class10 = pred_class10_1 <= pred_class10_2

    accuracy2 = (pred_class2 == data_target).float().mean().item()
    accuracy10 = (pred_class10 == data_target).float().mean().item()
    return accuracy2, accuracy10

def train_siamese(train_input, train_target, train_classes, loss_weights, ch1=64, ch2=64, fc=64, lr=0.25, epochs=15, mini_batch_size=100, verb=False):
    """
    Train siamese network
    """
    
    model = Siamese(ch1, ch2, fc)
    nb_samples = train_input.size(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Loop through epochs
    for e in range(epochs):
        # Loop through mini-batches
        for b in range(0, nb_samples, mini_batch_size):
            # Compute output
            pred2, (pred10_1, pred10_2) = model(train_input[b:b + mini_batch_size])

            # Compute losses
            # mnist classes losses
            loss10_1 = criterion(pred10_1, train_classes[b:b + mini_batch_size, 0])
            loss10_2 = criterion(pred10_2, train_classes[b:b + mini_batch_size, 1])
            loss10 = loss10_1 + loss10_2
            # target loss
            loss2 = criterion(pred2, train_target[b:b + mini_batch_size])

            # Lin. comb. of both losses
            tot_loss = loss_weights[0] * loss2 + loss_weights[1] * loss10

            # Optimize
            model.zero_grad()
            tot_loss.backward()
            optimizer.step()

        # After 5th epoch, print accuracy
        if e % 5  == 4 and verb:
            train_acc2, train_acc10 = siamese_accuracy(model, train_input, train_target)
            print(f"{e + 1}/{epochs} epochs:")
            print(f"2-classes train accuracy: {train_acc2 * 100:.2f}%")
            print(f"10-classes train accuracy: {train_acc10 * 100:.2f}%")
    
    return model

#%%
if __name__ == "__main__":
    try:
        # Load gridsearch results
        res2 = pd.read_pickle("./results/res2.pkl").reset_index(drop=True)
        res10 = pd.read_pickle("./results/res10.pkl").reset_index(drop=True)
        
        # Count the number of params for each model
        #res2["nb_params"] = res2["scenario"].apply(count_params)
        #res10["nb_params"] = res10["scenario"].apply(count_params)

        # Get best params from gridsearch
        params = []
        for df in [res2, res10]:
            best_idx = get_idx_best_acc(df)
            best_scen = df.loc[best_idx, "scenario"]
            params.append(best_scen)
            print("Best parameters loaded from gridsearch for model:\n" \
                    f"ch1: {best_scen[0]}, ch2: {best_scen[1]}, fc: {best_scen[2]}, lr: {best_scen[3]}, loss_weights: {best_scen[4]}")
    except Exception:
        # If there is any error loading the grisearch results
        print("Could not load grisearch results, hardcoding it")
        params = []
        params.append((64, 8, 64, 1, (1, 1)))  # 2-classes
        params.append((64, 16, 64, 0.1, (0, 1)))  # 10-classes

    # Generate data
    N, k = 1000, 10
    data = [generate_pair_sets(N) for _ in range(k)]

    # Train and test model on each of the dataset
    scores = []
    for i, d in enumerate(data):
        print(f"{i + 1}/{len(data)}")

        train_input, train_target, train_classes, test_input, test_target, test_classes = d

        # Standardize
        train_input, test_input = standardize(train_input), standardize(test_input)

        # Add baseline model

        # Train the 2-classes network
        model2 = train_siamese(train_input, train_target, train_classes, ch1=params[0][0], ch2=params[0][1], fc=params[0][2], epochs=25, mini_batch_size=100, lr=params[0][3], loss_weights=params[0][4])

        # Train the 10-classes network
        model10 = train_siamese(train_input, train_target, train_classes, ch1=params[1][0], ch2=params[1][1], fc=params[1][2], epochs=25, mini_batch_size=100, lr=params[1][3], loss_weights=params[1][4])

        # Once finished, print test accuracies
        test_acc2, _ = siamese_accuracy(model2, test_input, test_target)
        _, test_acc10 = siamese_accuracy(model10, test_input, test_target)
        scores.append((test_acc2, test_acc10))
        print(f"2-classes test accuracy: {test_acc2 * 100:.2f}%")
        print(f"10-classes test accuracy: {test_acc10 * 100:.2f}%")
    
    scores = torch.FloatTensor(scores) 
    test_acc2 = scores[:, 0] * 100
    test_acc10 = scores[:, 1] * 100
    print(f"Averages over {k} runs:")
    print(f"2-classees test_error {test_acc2.mean():.02f}% (std {test_acc2.std():.02f})")
    print(f"10-classees test_error {test_acc10.mean():.02f}% (std {test_acc10.std():.02f})")

# %%
"""
Averages over 10 runs:
2-classees test_error 91.82% (std 1.08)
10-classees test_error 97.35% (std 0.44)
"""