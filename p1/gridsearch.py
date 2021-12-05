#%%
import time

import torch

from dlc_practical_prologue import generate_pair_sets
from siamese import train_siamese, standardize, siamese_accuracy

#%%
def gridsearch_siamese(params_grid, n=2, epochs=15, verb=True):
    # Loop through each scenario, train the network n times
    # and compute accuracy
    # Return a dict with all scenarios and mean accuracies (+std)

    # Nb of pairs to generate
    N = 1000

    grid_results = {"siamese2": {}, "siamese10": {}}

    # Loop through each scenario
    for i, scenario in enumerate(params_grid):
        print(f"{i + 1}/{len(params_grid)} Scenario", scenario)
        t1 = time.perf_counter()

        # Set keys
        grid_results["siamese2"][scenario] = []
        grid_results["siamese10"][scenario]  = []
        ch1, ch2, fc, lr, lw = scenario

        # Train number of times to reduce variance
        scores = []
        for _ in range(n):
            # Get data
            train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
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
            
def sort_dict(results, reverse=True):
    return {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

#%%
if __name__ == "__main__":
    # Generate params to search
    params_grid = [
        (int(ch1), int(ch2), int(fc), lr, lw) 
            for ch1 in [2 ** exp for exp in (3, 4, 5, 6)]
            for ch2 in [2 ** exp for exp in (3, 4, 5, 6)]
            for fc in [2 ** exp for exp in (6, 7, 8, 9)]
            for lr in (0.001, 0.01, 0.1, 0.25, 1)
            for lw in [(1, 10 ** exp) for exp in (0, -0.5, -1.5, -2, -3, -4)] + [(1, 0),]
    ]

    # print(params_grid)

    # Test grid
    # params_grid = [(int(ch1), int(ch2), int(fc), lr, lw) 
    #             for ch1 in [2 ** exp for exp in (1, 2)]
    #             for ch2 in [2 ** exp for exp in (1, 2)]
    #             for fc in [2 ** exp for exp in (1, 2)]
    #             for lr in (0.001, 1)
    #             for lw in ((1, 10 ** -0.5), (0.5, 0.5))]

    results = gridsearch_siamese(params_grid, epochs=25, n=10)
    res2 = sort_dict(results["siamese2"])
    res10 = sort_dict(results["siamese10"])

# %%
# Testing

res = {
    "low": [(0.49, 0.09)],
    "medium": [(0.58, 0.1)],
    "high": [(0.59, 0.1)],
}

# Print results in descending order by accuracy
{k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

# %%
