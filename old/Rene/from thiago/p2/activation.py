from model import Module
import torch
import math


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.x = x
        return x.clamp(min=0)

    def backward(self, grad):
        #ds_dx = dReLU(self.x)
        ds_dx = (torch.sign(self.x) + 1)/2
        dl_dx = ds_dx*grad
        return dl_dx

    def reset_parameters(self):
        return

    '''def SGD(self, eta, momentum):
        return'''
    
    def parameters(self, eta, momentum):
        return 
    
    def update_parameters(self, eta):
        return
    
    def zero_gradient(self):
        return

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.x = x
        return x.tanh()

    def backward(self, grad):
        ds_dx = 4 * (self.x.exp() + self.x.mul(-1).exp()).pow(-2)
        dl_dx = ds_dx*grad
        return dl_dx

    def reset_parameters(self):
        return

    '''def SGD(self, eta, momentum):
        return'''
    
    def parameters(self, eta, momentum):
        return 
    
    def update_parameters(self, eta):
        return

    def zero_gradient(self):
        return


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, v, t):
        return (v - t).pow(2).sum()

    def backward(self, v, t):
        return 2 * (v - t)
