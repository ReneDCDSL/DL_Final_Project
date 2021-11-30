#%%
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

%load_ext autoreload
%autoreload 2

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
# Build different convnet architecture to predict target given input (and classes)
# Look into weight sharing and auxiliary losses

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 14, kernel_size=1)
        self.conv2 = nn.Conv2d(14, 28, kernel_size=1)
        self.fc1 = nn.Linear(5488, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        return x

class LeNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(2, 5, kernel_size=3),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(5, 16, kernel_size=3),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 64, kernel_size=2),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 84),
            nn.Tanh(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()  
        self.feats = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 84, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(84, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feats_cls = nn.Sequential(
            nn.Linear(192 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        x1 = self.feats(img1)
        x1 = self.avgpool(x1)
        x1 = self.feats_cls(x1)

        x2 = self.feats(img2)
        x2 = self.avgpool(x2)
        x2 = self.feats_cls(x2)

        x  = x1 * x2
        x = self.classifier(x)
        return x

class LeNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 25, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# %%
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
# With minibatches
scores = []
iters = 1 # for testing

for _ in range(iters):
    nb_epochs, mini_batch_size = 25, 100
    learning_rate = 1e-3
    model = Siamese()

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in tqdm(range(nb_epochs)):
        for b in range(0, train_input.size(0), mini_batch_size):
            # train model
            # print(train_input[b:b + mini_batch_size].size())
            # print(train_target[b:b + mini_batch_size].size())
            train_model(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size], loss_fn, optimizer)

    # Print accuracy once finished
    train_errs, test_errs = 0, 0
    for b in range(0, train_input.size(0), mini_batch_size):
        train_errs += count_errs(model, train_input[b:b + mini_batch_size], train_target[b:b + mini_batch_size]) 
        test_errs += count_errs(model, test_input[b:b + mini_batch_size], test_target[b:b + mini_batch_size])

    train_err_rate, test_err_rate = 0, 0
    train_err_rate += train_errs / train_input.size(0) * 100
    test_err_rate += test_errs / test_input.size(0) * 100
    scores.append((train_err_rate, test_err_rate))

    del model

scores = torch.FloatTensor(scores) 
train_err_rates = scores[:, 0] 
test_err_rates = scores[:, 1] 
print(f"train_error {train_err_rates.mean():.02f}% (std {train_err_rates.std():.02f})%")
print(f"test_error {test_err_rates.mean():.02f}% (std {test_err_rates.std():.02f})%")

# %%

