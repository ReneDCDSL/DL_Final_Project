class Module(object):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    def parameters(self):
        return []


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

'''    def SGD(self, eta, momentum):
        for module in self.module_lst:
            module.SGD(eta, momentum)'''
            
    def update_parameters(self, eta, momentum):
        for module in self.module_lst:
            module.update_parameters(eta, momentum)
            
    def parameters(self):
        lst = []
        for module in self.module_lst:
            lst.append(module.parameters)
        return lst

    def zero_gradient(self):
        for module in self.module_lst:
            module.zero_gradient()
        return

    def reset_parameters(self):
        for module in self.module_lst:
            module.reset_parameters()
