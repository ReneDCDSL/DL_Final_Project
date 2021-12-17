import torch
import math
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#generates train and test tensors with one_hot encoded targets as a mask target
def generate_disc_set(nb):
    # creating the circle in the middle of the points
    axis = torch.FloatTensor(1,2).uniform_(0.5, 0.5)
    r = 1/((2*math.pi)**0.5)

    train_input   =  torch.FloatTensor(nb, 2).uniform_(0,1)
    train_target  =  torch.FloatTensor(nb, 2)
    train_mask    =  torch.FloatTensor(nb, 1)
    test_input    =  torch.FloatTensor(nb, 2).uniform_(0,1)
    test_target   =  torch.FloatTensor(nb, 2)
    test_mask     =  torch.FloatTensor(nb, 1)

    for i in range(len(train_input)):
        a = abs((train_input[i] - axis).pow(2).sum(1).view(-1).pow(0.5))
        b = abs((test_input[i]  - axis).pow(2).sum(1).view(-1).pow(0.5))

        if a < r:
            train_target[i][0] = 0
            train_target[i][1] = 1
            train_mask[i]      = 1
        else:
            train_target[i][0] = 1
            train_target[i][1] = 0
            train_mask[i]      = 0

        if b < r:
            test_target[i][0] = 0
            test_target[i][1] = 1
            test_mask[i]      = 1
        else:
            test_target[i][0] = 1
            test_target[i][1] = 0
            test_mask[i]      = 0

    return train_input, train_target, test_input, test_target, test_mask, train_mask


# plot the points in the data set, uses a mask
def plot_set(data_input, data_mask):
    train_scatter = torch.cat((data_input, data_mask),1)
    train_scatter_false = train_scatter[train_scatter[:,2] == 0]
    train_scatter_true = train_scatter[train_scatter[:,2] == 1]

    plt.figure(figsize=(5, 5))
    plt.scatter(train_scatter_false[:,0], train_scatter_false[:,1], )
    plt.scatter(train_scatter_true[:,0], train_scatter_true[:,1], )
    plt.title("Data distribution")
    plt.show()
