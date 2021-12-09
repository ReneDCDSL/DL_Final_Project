from module import Module

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
        
    def params(self):
        return []
    
    def update_params(self, eta):
        return
    
    def reset_gradient(self):
        return

