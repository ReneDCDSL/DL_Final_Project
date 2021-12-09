import torch
import torch.nn as nn

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
