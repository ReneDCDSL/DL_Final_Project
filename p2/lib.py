import torch
torch.set_grad_enabled(False)

class Module:
    """
    Initial module abstract class
    """

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *gradswrtoutput):
        raise NotImplementedError
    
    def params(self):
        return []

class Linear(Module):
    """
    Linear layer
    """

    def __init__(self, in_feats, out_feats):
        self.weights = torch.empty((in_feats, out_feats))
        self.bias = torch.empty((1, out_feats))

    def forward(self, x):
        self.input = x
        output = x @ self.weights + self.bias
        return output

    def backward(self, dy):
        pass

    def params(self):
        return [self.weights, self.bias]

class SGD(Module):
    pass

class MSE(Module):
    pass

class Tanh(Module):
    pass

class ReLU(Module):
    pass