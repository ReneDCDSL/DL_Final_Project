class Module(object):
    def __init__(self):
        pass

    '''
    Compute forward pass from an input tensor and return a tensor
    or a tuple of tensors as output
    '''
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, grad):
        '''
        should get as input a tensor or a tuple of tensors containing the 
        gradient of the loss with respect to the module’s output, accumulate 
        the gradient wrt the parameters, and return a tensor or a tuple of
        tensors containing the gradient of the loss wrt the module’s input.
        '''
        raise NotImplementedError
        
    def params(self):
        '''
        param should return a list of pairs, each composed of a parameter tensor, and a gradient tensor
        of same size. This list should be empty for parameterless modules (e.g. ReLU).

        '''
        return []
        
    def reset_params(self):
        return
