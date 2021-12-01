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

print("Sanity check")
print(f"{train_input.shape=}")
print(f"{train_target.shape=}")
print(f"{train_classes.shape=}")
print(f"{test_input.shape=}")
print(f"{test_target.shape=}")
print(f"{test_classes.shape=}")

# %%
# Build different convnet architecture to predict target given input (and classes)
# Look into weight sharing and auxiliary losses

class LeNetLike(nn.Module):
    # For one channel
    def __init__(self):
        super().__init__()  
        self.feats = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.feats_cls = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feats(x)
        x = torch.flatten(x, 1)
        x = self.feats_cls(x)
        return x

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()  
        self.feats = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.feats_classifier = nn.Sequential(
            nn.Linear(64, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        x1 = self.feats(img1)
        x1 = torch.flatten(x1, 1)
        x1 = self.feats_classifier(x1)

        x2 = self.feats(img2)
        x2 = torch.flatten(x2, 1)
        x2 = self.feats_classifier(x2)

        x  = x1 * x2
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Siamese2(nn.Module):
    def __init__(self):
        super().__init__()  
        self.feats = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # self.feats_classifier = nn.Sequential(
        #     nn.Linear(64, 496),
        #     nn.ReLU(),
        #     nn.Linear(496,  120),
        #     nn.ReLU(),
        #     nn.Linear(120,  84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10),
        #     nn.ReLU()
        # )
        # self.feats_classifier = nn.Sequential(
        #     nn.Linear(64, 120),
        #     nn.ReLU(),
        #     nn.Linear(120, 84),
        #     nn.ReLU(),
        #     nn.Linear(84, 10),
        #     nn.ReLU()
        # )
        # test change
        self.feats_classifier = nn.Sequential(
            nn.Linear(64, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.ReLU()
        )

        out_features = 10
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        x1 = self.feats(img1)
        x1 = torch.flatten(x1, 1)
        x1 = self.feats_classifier(x1)

        x2 = self.feats(img2)
        x2 = torch.flatten(x2, 1)
        x2 = self.feats_classifier(x2)

        x  = x1 - x2  # if <= 0, then x1 <= x2 
        # x  = x1 * x2
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Siamese3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 2),  
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),    
            nn.MaxPool2d(2),  
            nn.Conv2d(128, 128, 2),
            nn.ReLU()
        )
        self.lin = nn.Sequential(
            nn.Linear(128, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 1),
            # nn.ReLU(),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(4096, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, img1, img2):
        x1 = self.conv(img1)    
        x1 = torch.flatten(x1, 1)
        x1 = self.lin(x1)

        x2 = self.conv(img2)
        x2 = torch.flatten(x2, 1)
        x2 = self.lin(x2)

        # x  = x1 * x2
        x  = x1 - x2  # if <= 0, then x1 <= x2 
        # x = torch.flatten(x, 1)
        x = self.out(x)
        return x

# %%
# General
# Doesn't work for siamese networks
def train_model(model, train_input, train_target, loss_fn, optimizer):
    # compute loss
    output = model(train_input)
    if isinstance(output, tuple):
        output, _ = output

    loss = loss_fn(output, train_target)

    # optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()

def count_errs(model, data_input, data_target):
    with torch.no_grad():
        output = model(data_input)
        if isinstance(output, tuple):
            output, _ = output
        _, preds = torch.max(output, 1)
    
    return (preds != data_target).sum().item()

# %%
# Siamese

def train_siamese(model, train_input, train_target, loss_fn, optimizer):
    # Get separate imgs
    img1 = train_input[:, 0, :, :]  # All first pair
    img1.unsqueeze_(1)
    img2 = train_input[:, 1, :, :]  # All second pair
    img2.unsqueeze_(1)

    # compute loss
    output = model(img1, img2)

    # To avoid warning
    train_target.unsqueeze_(1)
    loss = loss_fn(output, train_target.float())

    # optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()

def count_errs_siamese(model, data_input, data_target):
    # Get separate imgs
    img1 = data_input[:, 0, :, :]  # All first pair
    img1.unsqueeze_(1)
    img2 = data_input[:, 1, :, :]  # All second pair
    img2.unsqueeze_(1)

    model.eval()

    with torch.no_grad():
        probs = model(img1, img2)
    
    probs = torch.flatten(probs)
    # print(f"{probs=}")
    # print(f"predicted class {(probs > 0.5)}")
    # print(f"{data_target=}")

    # return (data_target != (probs > 0.5)).sum().item()
    return (data_target != (probs > 0.5)).sum().item()

# %%
# Test simple image classifier on MNIST
# Ignore 
if False:
    train_input, train_target, test_input, test_target = load_data(flatten=False, full=True)

    print("Sanity check")
    print(f"{train_input.shape=}")
    print(f"{train_target.shape=}")
    print(f"{test_input.shape=}")
    print(f"{test_target.shape=}")

    nb_epochs, mini_batch_size = 4, 100
    learning_rate = 1e-3
    model = LeNetLike()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in tqdm(range(nb_epochs)):
        for b in range(0, train_input.size(0), mini_batch_size):
            # train model
            train_model(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size], loss_fn, optimizer)

    # Print accuracy once finished
    train_errs, test_errs = 0, 0
    for b in range(0, train_input.size(0), mini_batch_size):
        train_errs += count_errs(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size]) 
        test_errs += count_errs(model, test_input[b:b + mini_batch_size], test_target[b:b + mini_batch_size])

    # Compute rates
    train_err_rate, test_err_rate = 0, 0
    train_err_rate += train_errs / train_input.size(0) * 100
    test_err_rate += test_errs / test_input.size(0) * 100
    print(f"train_error {train_err_rate:.02f}%")
    print(f"test_error {test_err_rate:.02f}%")
    # On full MNIST dataset, 15 epochs, 1.94% test error 

# %%
# Train with minibatches
scores = []
iters = 10  # for testing

for _ in range(iters):
    nb_epochs, mini_batch_size = 25, 100
    learning_rate = 1e-3
    model = Siamese3()

    # loss_fn = nn.BCELoss()  # Don't forget to change loss with model
    loss_fn = nn.BCEWithLogitsLoss()  # Siamese3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in tqdm(range(nb_epochs)):
        for b in range(0, train_input.size(0), mini_batch_size):
            # train model
            # print(train_input[b:b + mini_batch_size].size())
            # print(train_target[b:b + mini_batch_size].size())
            train_siamese(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size], loss_fn, optimizer)

    # Print accuracy once finished
    train_errs, test_errs = 0, 0
    for b in range(0, train_input.size(0), mini_batch_size):
        train_errs += count_errs_siamese(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size]) 

        test_errs += count_errs_siamese(model, test_input[b:b + mini_batch_size], test_target[b:b + mini_batch_size])

    train_err_rate, test_err_rate = 0, 0
    train_err_rate += train_errs / train_input.size(0) * 100
    test_err_rate += test_errs / test_input.size(0) * 100
    scores.append((train_err_rate, test_err_rate))

    del model  # to retrain from scratch on next iteration

scores = torch.FloatTensor(scores) 
train_err_rates = scores[:, 0] 
test_err_rates = scores[:, 1] 
print(f"train_error {train_err_rates.mean():.02f}% (std {train_err_rates.std():.02f}%)")
print(f"test_error {test_err_rates.mean():.02f}% (std {test_err_rates.std():.02f}%)")

# %%
# Some tests
randid = torch.randint(1001, (1,)).item()
x = train_input[randid]
x1 = x[0]
x2 = x[1]

x1 = torch.flatten(x1)
l1 = nn.Linear(14 * 14, 1)(x1)
prob = nn.Sigmoid()(l1)
y = train_target[randid]

print("true class", y.item())
print("prob", prob.item())
print("predicted class (0=False, 1=True)", 1 if (prob > 0.5).item() else 0)
print("correct", y == (prob > 0.5))

# %%
