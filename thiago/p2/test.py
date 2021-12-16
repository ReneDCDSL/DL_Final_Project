from model import Module, Sequential
from activation import ReLU, Tanh, MSELoss
from linear import Linear
from disc_generator import generate_disc_set, plot_set
from train import train_model

import torch

#turn off autograd
torch.set_grad_enabled( False )

#generate de disc
train_input, train_target_one_hot, test_input, test_target_one_hot, test_target, train_target = generate_disc_set(2000)


#plot the train data set distribution
#plot_set(train_input, train_target)


# define the constants
input_units = 2
output_units = 1
hidden = 25
nb_epochs = 100
mini_batch_size = 100
eta = 1e-3
momentum = 0.9


#create the model
mini_model = Sequential(
                        Linear(input_units, hidden),  ReLU(),
                        Linear(hidden, hidden),       ReLU(),
                        Linear(hidden, hidden),       Tanh(),
                        Linear(hidden, output_units), Tanh())


# print the models structure
n = 0
for i in mini_model.parameters():
    n = n + 1
    try:
       print(f"w{n}: {i[0].size()}  b{n}: {i[1].size()}")
    except:
        n =n-1
        continue


# trains the model and give train and test results
#if Show graphs = True, the predicted distribution will be ploted every 3 epochs
train_model(mini_model, train_input, train_target,
                        test_input,  test_target,
                        nb_epochs,   mini_batch_size,
                        eta = eta,   momentum = momentum,
                        show_graphs = False, show_steps = True)
