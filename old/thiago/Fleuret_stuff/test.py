from sequential import Sequential
from linear import Linear
from tanh import Tanh
import helper
from importlib import reload
from helper import *

# torch.random.manual_seed(2949543082284623525)

# Definition of constants
input_units = 2
output_units = 2
hidden_units = 25
nb_epochs = 100
test_size = 1000
train_size = 1000

# Generate train / test set
train_input, train_target = generate_disc_set(train_size)
test_input, test_target = generate_disc_set(test_size)
train_target_one_hot = one_hot(train_target)
test_target_one_hot = one_hot(test_target)
train_target_one_hot = one_hot(train_target)
test_target_one_hot = one_hot(test_target)

# Initialize network
network = Sequential(
            Linear(input_units, hidden_units),
            Tanh(),
            Linear(hidden_units, hidden_units),
            Tanh(),
            Linear(hidden_units, output_units),
            Tanh())

# Train the model
mini_batch_size = 100

for i in range(100):
    train_model(network, train_input, train_target_one_hot, nb_epochs, mini_batch_size)
    print("Train accuracy: ", round(compute_accuracy(network, train_input, train_target, mini_batch_size), 2))
    print("Test accuracy:", round(compute_accuracy(network, test_input, test_target, mini_batch_size), 2))

