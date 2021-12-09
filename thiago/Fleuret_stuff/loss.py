from module import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, v, t):
        return (v - t).pow(2).sum()
    
    def backward(self, v, t):
        return 2 * (v - t)
