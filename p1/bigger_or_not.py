"""
Simple MLP to classify if the first member of a pair is smaller or equal to the second member.

if pair[0] <= pair[1]:
    predicted_class = 0
else:
    predicted_class = 1
"""

#%%
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

%load_ext autoreload
%autoreload 2

from dlc_practical_prologue import load_data
from dlc_practical_prologue import generate_pair_sets

# %%
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

print("Sanity check")
print(f"{train_input.shape=}")
print(f"{train_target.shape=}")
print(f"{train_classes.shape=}")
print(f"{test_input.shape=}")
print(f"{test_target.shape=}")
print(f"{test_classes.shape=}")

# %%
class BiggerOrNot(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# %%
def train(model, train_classes, train_target, loss_fn, optimizer):
    # compute loss
    output = model(train_classes.float())

    # To avoid warning
    loss = loss_fn(output, train_target.unsqueeze(1).float())

    # optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()

def count_errs(model, data_classes, data_target):
    model.eval()

    with torch.no_grad():
        probs = model(data_classes.float())
    
    probs = torch.flatten(probs)
    # print(f"{data_classes=}")
    # print(f"{probs=}")
    # print(f"predicted class {(probs > 0.5)}")
    # print(f"{data_target=}")

    return (data_target != (probs > 0.5)).sum().item()

# %%
iters = 25
scores = []

for _ in range(iters):
    # Regenerate random pairs
    N = 1000
    pairs = generate_pair_sets(N)

    # Train
    train_target = pairs[1]
    train_classes = pairs[2]

    # Test
    test_target = pairs[4]
    test_classes = pairs[5]

    nb_epochs, lr = 15, 1e-3
    model = BiggerOrNot()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(nb_epochs):
        train(model, train_classes, train_target, loss_fn, optimizer)

    # Compute errors
    # Training set
    train_errs = count_errs(model, train_classes, train_target) 
    test_errs = count_errs(model, test_classes, test_target)

    train_err_rate = train_errs / train_classes.size(0) * 100
    test_err_rate = test_errs / test_classes.size(0) * 100
    scores.append((train_err_rate, test_err_rate))
    # print(f"train_error {train_err_rate:.02f}%")
    # print(f"test_error {test_err_rate:.02f}%")

    del model

scores = torch.FloatTensor(scores) 
train_err_rates = scores[:, 0] 
test_err_rates = scores[:, 1] 
print(f"train_error {train_err_rates.mean():.02f}% (std {train_err_rates.std():.02f}%)")
print(f"test_error {test_err_rates.mean():.02f}% (std {test_err_rates.std():.02f}%)")
# %%

# %%
