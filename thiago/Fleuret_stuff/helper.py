import torch
from torch import Tensor
import math
from loss import MSELoss


def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    return input, target


def one_hot(target):
    one_hot_target = torch.zeros(target.shape[0], 2)
    for i in range(one_hot_target.shape[0]):
        if target[i] == 0:
            one_hot_target[i, 0] = 1
            one_hot_target[i, 1] = -1
        else:
            one_hot_target[i, 1] = 1
            one_hot_target[i, 0] = -1
    return one_hot_target


def compute_accuracy(model, input, target, mini_batch_size):
    nb_error = 0
    for b in range(0, input.size(0), mini_batch_size):
        output = model.forward(input.narrow(0, b, mini_batch_size))
        pred = output.max(1)[1]
        batch_error = (pred - target.narrow(0, b, mini_batch_size)).abs().sum()
        nb_error += batch_error
    return 100 * (1 - nb_error.item() / len(target))


def train_model(model, train_input, train_target, nb_epochs, mini_batch_size, criterion=MSELoss(), eta=1e-3):
    model.reset_params()
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            # forward pass
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss += loss.item()

            # backward pass
            model.reset_gradient()
            model.backward(criterion.backward(output, train_target.narrow(0, b, mini_batch_size)))
            model.update_params(eta)
