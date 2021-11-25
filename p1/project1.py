#%%
import torch
from dlc_practical_prologue import generate_pair_sets

#%%
N = 1000
pairs = generate_pair_sets(N)

# Train
train_input = pairs[0]
train_target = pairs[1]
train_classes = pairs[2]

# Test
test_input = pairs[3]
test_target = pairs[4]
test_classes = pairs[5]

# %%
print("Sanity check")
print(f"{train_input.shape=}")
print(f"{train_target.shape=}")
print(f"{train_classes.shape=}")
print(f"{test_input.shape=}")
print(f"{test_target.shape=}")
print(f"{test_classes.shape=}")

# %%
# Check if task is understood
# Target == 1 if pair[0] <= pair[1] 

# Return first element and index of element that fulfills the condition
for i, pair in enumerate(train_classes):
    if pair[0] <= pair[1]:
        print("element index", i, "values", pair.tolist())
        print("element target", train_target[i].item())
        break

# %%
# Build different convnet architecture to predict target given input (and classes?)
# Look into weight sharing and auxiliary losses
