from module import Module


def sigmoid(x):
    return 1/(1 + x.mul(-1).exp())

    
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.x = 0
            
    def forward(self, x):
        self.x = x
        return sigmoid(self.x)
        
    def backward(self, grad):
        ds_dx = sigmoid(self.x)*(1-sigmoid(self.x))
        dl_dx = ds_dx*grad
        return dl_dx
