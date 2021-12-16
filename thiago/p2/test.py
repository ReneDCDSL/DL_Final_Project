from model import Module, Sequential
from activation import ReLU, Tanh, MSELoss
from linear import Linear
from disc_generator import generate_disc_set, plot_set
from train import train_model

train_input, train_target_one_hot, test_input, test_target_one_hot, test_target, train_target = generate_disc_set(2000)



#plot_set(train_input, train_target)

input_units = 2
output_units = 1
hidden = 25
nb_epochs = 100
mini_batch_size = 100
eta = 1e-3
momentum = 0.9


mini_model = Sequential(
                        Linear(input_units, hidden),  ReLU(),
                        Linear(hidden, hidden),       ReLU(),
                        Linear(hidden, hidden),       Tanh(),
                        Linear(hidden, output_units), Tanh())

#n = 0
#for i in mini_model.parameters():
    #n = n + 1
    #print(f"w{n}: {i[0].size()}  b{n}: {i[1].size()}")



train_model(mini_model, train_input, train_target,
                        test_input,  test_target,
                        nb_epochs,   mini_batch_size,
                        eta = eta,   momentum = momentum,
                        show_graphs = False, show_steps = True)
