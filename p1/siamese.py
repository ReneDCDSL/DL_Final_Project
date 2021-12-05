#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dlc_practical_prologue import generate_pair_sets

# %%
class Siamese(nn.Module):
    def __init__(self, ch1=64, ch2=64, fc=64):
        super().__init__()

        self.ch1 = ch1
        self.ch2 = ch2
        self.fc = fc

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.ch1, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(self.ch1),
            nn.Conv2d(self.ch1, self.ch2, 6),
            nn.ReLU(),
            nn.BatchNorm2d(self.ch2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.ch2, self.fc),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc),
            nn.Linear(self.fc, 10),
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
def standardize(data):
    mean, std = data.mean(), data.std()
    return (data - mean)/std

def siamese_accuracy(model, data_input, data_target):
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

def train_siamese(train_input, train_target, train_classes, loss_weights, ch1=64, ch2=64, fc=64, lr=0.25, epochs=15, mini_batch_size=100, verb=True):
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
            # mnist classees losses
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
    # Train the network
    N = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
    train_input, test_input = standardize(train_input), standardize(test_input)
    loss_weights = (1, 10 ** -0.5)

    model = train_siamese(train_input, train_target, train_classes, loss_weights=loss_weights, epochs=25, mini_batch_size=100, lr=0.1)

    # Once finished, print test accuracy
    test_acc2, test_acc10 = siamese_accuracy(model, test_input, test_target)
    print("\nTest accuracy:")
    print(f"2-classes test accuracy: {test_acc2 * 100:.2f}%")
    print(f"10-classes test accuracy: {test_acc10 * 100:.2f}%")

# %%
