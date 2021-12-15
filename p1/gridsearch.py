import os
import time

import pickle as pkl
import torch

from dlc_practical_prologue import generate_pair_sets
from train_test import baseline_accuracy, siamese_accuracy
from train_test import train_baseline, train_siamese 
from train_test import standardize

def gridsearch_siamese(params_grid, n=2, epochs=15, verb=True):
    """
    Loop through each scenario, train the network n times
    and compute accuracy
    Return a dict with all scenarios and mean accuracies (+std)
    """

    # Nb of pairs to generate
    N = 1000

    grid_results = {"baseline": {}, "siamese2": {}, "siamese10": {}}

    # Loop through each scenario
    for i, scenario in enumerate(params_grid):
        print(f"{i + 1}/{len(params_grid)} Scenario", scenario)
        t1 = time.perf_counter()

        # Set keys and params
        grid_results["siamese2"][scenario] = []
        grid_results["siamese10"][scenario]  = []
        ch1, ch2, fc, lr, lw = scenario

        # Train number of times to reduce variance
        scores = []
        for _ in range(n):
            # Get data
            train_input, train_target, train_classes, test_input, test_target, _ = generate_pair_sets(N)
            train_input, test_input = standardize(train_input), standardize(test_input)

            # Train the network wrt scenario
            model = train_siamese(train_input, train_target, train_classes, loss_weights=lw, ch1=ch1, ch2=ch2, fc=fc, lr=lr, epochs=epochs, verb=False)

            # Compute test accuracy
            test_acc2, test_acc10 = siamese_accuracy(model, test_input, test_target)
            scores.append((test_acc2, test_acc10))

        # At the end of each scenario, get mean accuracy and std
        # of the runs
        scores = torch.FloatTensor(scores)
        test_acc2, test_acc10 = scores[:, 0], scores[:, 1]  
        
        grid_results["siamese2"][scenario].append((test_acc2.mean().item(), test_acc2.std().item()))
        grid_results["siamese10"][scenario].append((test_acc10.mean().item(), test_acc10.std().item()))

        if verb:
            print(f"2-classes test accuracy: {test_acc2.mean() * 100:.2f}% (+/- {test_acc2.std() * 100:.2f}%)")
            print(f"10-classes test accuracy: {test_acc10.mean() * 100:.2f}% (+/- {test_acc10.std() * 100:.2f}%)")
            print(f"Time {time.perf_counter() - t1:.2f}s")

    return grid_results

def gridsearch_baseline(params_grid, epochs=15, n=2, verb=True):
    """
    Loop through each scenario, train the network n times
    and compute accuracy
    Return a dict with all scenarios and mean accuracies (+std)
    """

    # Generate data
    N = 1000
    data = [generate_pair_sets(N) for _ in range(n)]

    grid_results = {"baseline": {}}

    # Loop through each scenario
    for i, scenario in enumerate(params_grid):
        print(f"{i + 1}/{len(params_grid)} Scenario", scenario)
        t1 = time.perf_counter()

        # Set keys and params
        grid_results["baseline"][scenario] = []
        ch1, ch2, fc1, fc2, lr, standard = scenario

        # Train number of times
        scores = []
        for d in data:
            # Get data
            train_input, train_target, train_classes, test_input, test_target, test_classes = d
            if standard:
                train_input, test_input = standardize(train_input), standardize(test_input)

            # Train the network wrt scenario
            _, model = train_baseline(train_input, train_target, test_input, test_target, ch1=ch1, ch2=ch2, fc1=fc1, fc2=fc2, lr=lr, epochs=epochs, verb=True, eval=False)

            # Compute test accuracy
            test_acc = baseline_accuracy(model, test_input, test_target)
            scores.append(test_acc)

        # At the end of each scenario, get mean accuracy and std
        # of the runs
        scores = torch.FloatTensor(scores)
        
        grid_results["baseline"][scenario].append((scores.mean().item(), scores.std().item()))

        if verb:
            print(f"baseline test accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
            print(f"Time {time.perf_counter() - t1:.2f}s")


    return grid_results

def sort_dict(results, reverse=True):
    """
    Sort result dict by highest accuracy
    """
    
    return {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

def run_gridsearch(params_grid, mode="baseline", epochs=15, n=5, save=True):
    """
    Run gridsearch for a given parameters grid and a given model (baseline or siamese)
    """
    
    if mode == "siamese":
        results = gridsearch_siamese(params_grid, epochs=epochs, n=n)
        res2 = sort_dict(results["siamese2"])
        res10 = sort_dict(results["siamese10"])
        
        if save:
            with open("./results/gridsearch_res2.tmp.pkl", "wb") as f:
                 pkl.dump(res2, f)
            with open("./results/gridsearch_res10.tmp.pkl", "wb") as f:
                 pkl.dump(res10, f)
        return res2, res10
    else:
        results = gridsearch_baseline(params_grid, epochs=epochs, n=n)
        res_dict = sort_dict(results["baseline"])
        
        if save:
            with open("./results/gridsearch_res_base.tmp.pkl", "wb") as f:
                pkl.dump(res_dict, f)
        return res_dict
    
if __name__ == "__main__":
    os.environ["PYTORCH_DATA_DIR"] = "/home/olivier/Documents/projects/courses/DL/data"

    # Siamese
    print("Gridsearching for siamese")
    
    # Generate params to search
    params_grid = [
        (int(ch1), int(ch2), int(fc), lr, lw) 
            for ch1 in [2 ** exp for exp in (3, 4, 5, 6)]
            for ch2 in [2 ** exp for exp in (3, 4, 5, 6)]
            for fc in [2 ** exp for exp in (6, 7, 8, 9)]
            for lr in (0.001, 0.01, 0.1, 0.25, 1)
            for lw in [(1, 10 ** exp) for exp in (0, -0.5, -1.5, -2, -3, -4)] + [(1, 0), (0, 1)]
    ]
    run_gridsearch(params_grid, mode="siamese", epochs=25, n=10)

    # Baseline
    print("Gridsearching for baseline")
    
    # Generate params to search
    params_grid = [
       (int(ch1), int(ch2), int(fc1), int(fc2), lr, standard) 
           for ch1 in [2 ** exp for exp in (3, 4, 5, 6)]
           for ch2 in [2 ** exp for exp in (3, 4, 5, 6)]
           for fc1 in [2 ** exp for exp in (6, 7, 8, 9)]
           for fc2 in [2 ** exp for exp in (6, 7, 8, 9)]
           for lr in (0.001, 0.01, 0.1, 0.25, 1)
           for standard in [True, False]
    ]
    
    # Dummy grid for baseline
    # params_grid = [
    #     (int(ch1), int(ch2), int(fc1), int(fc2), lr, standard) 
    #         for ch1 in [16, 32]
    #         for ch2 in [16, 32]
    #         for fc1 in [12, 64]
    #         for fc2 in [12, 128]
    #         for lr in (0.001, 0.01,)
    #         for standard in [False]
    # ]
    
    run_gridsearch(params_grid, mode="baseline", epochs=25, n=10)
