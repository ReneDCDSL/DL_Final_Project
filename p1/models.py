#%%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlc_practical_prologue import generate_pair_sets


class Baseline(nn.Module):  
    def __init__(self, ch1, ch2, fc1, fc2):
        super().__init__()

        self.ch1 = ch1
        self.ch2 = ch2
        self.fc1 = fc1
        self.fc2 = fc2

        self.conv = nn.Sequential(
            nn.Conv2d(2, ch1, kernel_size=4),
            nn.MaxPool2d(4, 1),
            nn.ReLU(),
            nn.BatchNorm2d(ch1),
            nn.Conv2d(ch1, ch2, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm2d(ch2),
        )

        self.fc = nn.Sequential(
            nn.Linear(ch2 * 5 * 5, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Siamese(nn.Module):
    def __init__(self, ch1=64, ch2=64, fc=64):
        super().__init__()

        self.ch1 = ch1
        self.ch2 = ch2
        self.fc = fc

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch1, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(ch1),
            nn.Conv2d(ch1, ch2, 6),
            nn.ReLU(),
            nn.BatchNorm2d(ch2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(ch2, fc),
            nn.ReLU(),
            nn.BatchNorm1d(fc),
            nn.Linear(fc, 10),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(20, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x1 = self.conv(x1)
        x1 = self.fc1(x1.view(-1, self.ch2))
        x2 = self.conv(x2)
        x2 = self.fc1(x2.view(-1, self.ch2))

        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(x)
        return x, (x1, x2)

#%% 
def train_baseline(train_input,train_target, ch1, ch2, fc1, fc2, mini_batch_size=100, epochs=20, lr=1e-1, mse=True, verb=True):
    train_acc_loss_list = []
    train_acc = []
    test_acc = []

    model = Baseline(ch1, ch2, fc1, fc2)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if mse == True:
        criterion = nn.MSELoss()
        F.one_hot(train_target).float()
        # F.one_hot(test_target).float()
    else:
        criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            if mse == True:
                loss = criterion(output, F.one_hot(train_target).float().narrow(0, b, mini_batch_size))
            else:
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc_loss_list.append(acc_loss)

        #Check the accuracy in the test set
        train_acc.append(baseline_accuracy(model, train_input, train_target))
        # test_acc.append(baseline_accuracy(model, test_input, test_target))

        return train_acc_loss_list, train_acc, test_acc, model

def baseline_accuracy(model, data_input, data_target):
    with torch.no_grad():
        output = model(data_input)
        _, preds = torch.max(output, 1)
    
    return (preds == data_target).float().mean().item()

#%%
if __name__ == "__main__":
    # Configuration
    os.environ["PYTORCH_DATA_DIR"] = "/home/olivier/Documents/projects/courses/DL/data"

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

    _, train_acc, test_acc, model = train_baseline(train_input, train_target, test_input, test_target, ch1=32, ch2=24, fc1=128, fc2=32, lr=1e-1, mse=True)

    print("Test accuracy", baseline_accuracy(model, test_input, test_target))
    
# %%

# %%
