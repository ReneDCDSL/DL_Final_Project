from module import Module
import torch


def dReLU(x):
    s = x.clone()
    s[x>0] = 1
    s[x<=0] = 0
    return s


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
    
    def update_params(self, eta):
        return
    
    def reset_gradient(self):
        return
