import math
import torch
from activation import MSELoss
import matplotlib.pyplot as plt
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#% matplotlib inline


def train_model(model, train_input, train_target,
                test_input, test_target,
                nb_epochs, mini_batch_size,
                criterion=MSELoss(),
                eta = 1e-3, momentum = 0,
                show_graphs = False, show_steps = True):

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            # forward pass
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss += loss.item()

            model.zero_gradient()
            model.backward(criterion.backward(output, train_target.narrow(0, b, mini_batch_size)))
            model.SGD(eta, momentum)

            train_error = 0
            test_error = 0
            for c in range(0, train_input.size(0), mini_batch_size):
                output = model.forward(train_input.narrow(0, c, mini_batch_size))
                pred   = torch.where(output**2 < 0.25, 0, 1)
                train_error = train_error + (pred != train_target.narrow(0, c, mini_batch_size)).sum()

                output = model.forward(test_input.narrow(0, c, mini_batch_size))
                pred   = torch.where(output**2 < 0.25, 0, 1)
                test_error = test_error + (pred != test_target.narrow(0, c, mini_batch_size)).sum()

            train_error_rate = (train_error/train_input.size(0)) * 100
            test_error_rate = (test_error/test_input.size(0)) * 100


        if show_graphs == True:
            if e % 3 ==  0:
                    output = model.forward(train_input)
                    pred   = torch.where(output < 0.5, 0, 1)
                    scatter = torch.cat((train_input, pred),1)
                    scatter_false = scatter[scatter[:,2] == 0]
                    scatter_true  = scatter[scatter[:,2] == 1]

                    plt.figure(figsize=(5, 5))
                    plt.scatter(scatter_false[:,0], scatter_false[:,1], )
                    plt.scatter(scatter_true[:,0], scatter_true[:,1], )




        if show_steps == True:
            print("epoch: {}, loss: {:.02f}, train error {:.02f}%, test error {:.02f}% ".format(e, sum_loss, train_error_rate, test_error_rate))

        if test_error == 0:
            print("I already know what is a circle")
            break

        if e > 20000:
            print("Master why did you make my so dumb?")
            break

    if show_steps == False:
        print("epoch: {}, loss: {:.02f}, train error {:.02f}%, test error {:.02f}% ".format(e, sum_loss, train_error_rate, test_error_rate))
        #output = model.forward(train_input)
