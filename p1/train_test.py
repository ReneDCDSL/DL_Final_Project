#%%
import time

import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlc_practical_prologue import generate_pair_sets
from models import Baseline, Siamese
from utils import count_params, get_idx_best_acc, standardize

def baseline_accuracy(model, data_input, data_target):
    with torch.no_grad():
        output = model(data_input)
        _, preds = torch.max(output, 1)
    
    return (preds == data_target).float().mean().item()

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

def train_baseline(train_input, train_target, test_input, test_target, ch1, ch2, fc1, fc2, mini_batch_size=100, epochs=20, lr=1e-1, verb=True):

    info = {"train_acc_loss": [], "test_acc_loss": [], "train_acc": [], "test_acc": []}

    model = Baseline(ch1, ch2, fc1, fc2)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        train_acc_loss, test_acc_loss = 0, 0

        model.train()
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            train_acc_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        info["train_acc_loss"].append(train_acc_loss)

        # Eval mode
        model.eval()
        with torch.no_grad():
            # Compute test loss
            for b in range(0, test_input.size(0), mini_batch_size):
                output = model(test_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, test_target.narrow(0, b, mini_batch_size))
                test_acc_loss += loss.item()
            info["test_acc_loss"].append(test_acc_loss)

            #Append accuracies
            info["train_acc"].append(baseline_accuracy(model, train_input, train_target))
            info["test_acc"].append(baseline_accuracy(model, test_input, test_target))

    return info, model
        
def train_siamese(train_input, train_target, train_classes, loss_weights, ch1=64, ch2=64, fc=64, lr=0.25, epochs=15, mini_batch_size=100, verb=False):

    info = {"train_acc_loss": [], "test_acc_loss": [], "train_acc2": [], "train_acc10": [], "test_acc2": [], "test_acc10": []}
    
    model = Siamese(ch1, ch2, fc)
    nb_samples = train_input.size(0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Loop through epochs
    for e in range(epochs):
        train_acc_loss, test_acc_loss = 0, 0

        # Train loop
        model.train()
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
            train_acc_loss += tot_loss.item()

            # Optimize
            model.zero_grad()
            tot_loss.backward()
            optimizer.step()

        info["train_acc_loss"].append(train_acc_loss)

        # Eval loop
        model.eval()
        with torch.no_grad():
            for b in range(0, nb_samples, mini_batch_size):
                # Compute output
                pred2, (pred10_1, pred10_2) = model(train_input[b:b + mini_batch_size])

                # Compute test losses
                # mnist classes losses
                loss10_1 = criterion(pred10_1, train_classes[b:b + mini_batch_size, 0])
                loss10_2 = criterion(pred10_2, train_classes[b:b + mini_batch_size, 1])
                loss10 = loss10_1 + loss10_2
                # target loss
                loss2 = criterion(pred2, train_target[b:b + mini_batch_size])

                # Lin. comb. of both losses
                tot_loss = loss_weights[0] * loss2 + loss_weights[1] * loss10
                test_acc_loss += tot_loss.item()

            info["test_acc_loss"].append(test_acc_loss)

            #Append accuracies
            train_acc2, train_acc10 = siamese_accuracy(model, train_input, train_target)
            test_acc2, test_acc10 = siamese_accuracy(model, test_input, test_target)
            info["train_acc2"].append(train_acc2)
            info["train_acc10"].append(train_acc10)
            info["test_acc2"].append(test_acc2)
            info["test_acc10"].append(test_acc10)

        # After 5th epoch, print accuracy
        if e % 5  == 4 and verb:
            print(f"{e + 1}/{epochs} epochs:")
            print(f"2-classes train accuracy: {train_acc2 * 100:.2f}%")
            print(f"10-classes train accuracy: {train_acc10 * 100:.2f}%")
    
    return info, model

#%%
if __name__ == "__main__":
    t1 = time.perf_counter()

    try:
        # Load gridsearch results
        resbase = pd.read_pickle("./results/res_base.pkl").reset_index(drop=True)
        res2 = pd.read_pickle("./results/res2.pkl").reset_index(drop=True)
        res10 = pd.read_pickle("./results/res10.pkl").reset_index(drop=True)

        # Get best params from gridsearch
        params = []
        for i, df in enumerate([resbase, res2, res10]):
            best_idx = get_idx_best_acc(df)
            best_scen = df.loc[best_idx, "scenario"]
            params.append(best_scen)
            if i == 0:
                print("Best parameters loaded from gridsearch for baseline:\n" \
                    f"ch1: {best_scen[0]}, ch2: {best_scen[1]}, fc1: {best_scen[2]}, fc2: {best_scen[3]}, lr: {best_scen[4]}, standardize: {best_scen[5]}")                
            else:
                print("Best parameters loaded from gridsearch for model:\n" \
                    f"ch1: {best_scen[0]}, ch2: {best_scen[1]}, fc: {best_scen[2]}, lr: {best_scen[3]}, loss_weights: {best_scen[4]}")
    except Exception as e:
        # If there is any error loading the gridsearch results
        print("Could not load gridsearch results, hardcoding it")
        params = []
        params.append((64, 64, 128, 128, 0.1, False))  # baseline
        params.append((64, 8, 64, 1, (1, 1)))  # 2-classes
        params.append((64, 16, 64, 0.1, (0, 1)))  # 10-classes

    # Generate data
    # !! DON'T FORGET TO EDIT K !!
    N, k = 1000, 10
    data = [generate_pair_sets(N) for _ in range(k)]

    # Train and test model on each of the dataset
    scores = []
    info_models = {"baseline": [], "siamese2": [], "siamese10": []}
    for i, d in enumerate(data):
        print(f"{i + 1}/{len(data)}")

        train_input, train_target, train_classes, test_input, test_target, test_classes = d

        # Standardize
        train_input, test_input = standardize(train_input), standardize(test_input)

        # Train baseline model
        info_baseline, modebase = train_baseline(train_input, train_target, test_input, test_target, ch1=params[0][0], ch2=params[0][1], fc1=params[0][2], fc2=params[0][3], epochs=25, mini_batch_size=100, lr=params[0][4])
        info_models["baseline"].append(info_baseline)

        # Train the 2-classes network
        info_siam2, model2 = train_siamese(train_input, train_target, train_classes, ch1=params[1][0], ch2=params[1][1], fc=params[1][2], epochs=25, mini_batch_size=100, lr=params[1][3], loss_weights=params[1][4])
        info_models["siamese2"].append(info_siam2)

        # Train the 10-classes network
        info_siam10, model10 = train_siamese(train_input, train_target, train_classes, ch1=params[2][0], ch2=params[2][1], fc=params[2][2], epochs=25, mini_batch_size=100, lr=params[2][3], loss_weights=(0, 1))
        info_models["siamese10"].append(info_siam10)

        # Once finished, print test accuracies
        base_acc = baseline_accuracy(modebase, test_input, test_target)
        test_acc2, _ = siamese_accuracy(model2, test_input, test_target)
        _, test_acc10 = siamese_accuracy(model10, test_input, test_target)
        scores.append((base_acc, test_acc2, test_acc10))
        print(f"baseline test accuracy: {base_acc * 100:.2f}%")
        print(f"2-classes test accuracy: {test_acc2 * 100:.2f}%")
        print(f"10-classes test accuracy: {test_acc10 * 100:.2f}%")
    
    scores = torch.FloatTensor(scores) 
    base_acc = scores[:, 0] * 100
    test_acc2 = scores[:, 1] * 100
    test_acc10 = scores[:, 2] * 100

    print(f"Averages over {k} runs:")
    print(f"baseline test accuracy {base_acc.mean():.02f}% (std {base_acc.std():.02f})")
    print(f"2-classes test accuracy {test_acc2.mean():.02f}% (std {test_acc2.std():.02f})")
    print(f"10-classes test accuracy {test_acc10.mean():.02f}% (std {test_acc10.std():.02f})")

    # Save information about models
    with open("./results/info_models.pkl", "wb") as f:
        pickle.dump(info_models, f)

    print(f"Done in {time.perf_counter() - t1:.2f}s")

"""
Averages over 10 runs:
baseline test accuracy 83.23% (std 1.17)
2-classes test accuracy 91.50% (std 1.65)
10-classes test accuracy 97.23% (std 0.57)
"""
