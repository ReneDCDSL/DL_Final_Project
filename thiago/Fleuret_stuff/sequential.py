from module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.module_lst = []
        for module in modules:
            self.module_lst.append(module)
    
    def forward(self, x):
        for module in self.module_lst:
            x = module.forward(x)
        return x
        
    def backward(self, grad):
        for module in reversed(self.module_lst):
            grad = module.backward(grad)
        return grad
    
    def update_params(self, eta):
        for module in self.module_lst:
            module.update_params(eta)
            
    def params(self):
        lst = []
        for module in self.module_lst:
            lst.append(module.params())
        return lst
    
    def reset_gradient(self):
        for module in self.module_lst:
            module.reset_gradient()
        return
    
    def reset_params(self):
        for module in self.module_lst:
            module.reset_params()
