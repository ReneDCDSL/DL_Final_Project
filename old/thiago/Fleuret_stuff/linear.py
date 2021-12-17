from module import Module
import torch
from torch.nn.init import xavier_normal_, xavier_normal


class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.epsilon = 1e-3
        self.x = 0

        # Initialize weights
        self.w = xavier_normal_(torch.empty(self.dim_out, self.dim_in))
        self.b = torch.empty(self.dim_out).normal_(0, self.epsilon)

        # Initialize gradient
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())
    
    def forward(self, x):
        self.x = x
        return self.x.mm(self.w.t()) + self.b


    def backward(self, grad):
        ds_dx = self.w.t()

        # do the same for every batch (batch dim becomes 1)
        dl_dx = ds_dx @ grad.t()

        # put batch dim back to 0
        dl_dx = dl_dx.t()

        # sum over all the outer product between (grad_1 * x_1^T) (_1 denotes not using mini-batches)
        self.dl_dw.add_(grad.t() @ self.x)

        # sum over the batch
        self.dl_db.add_(grad.sum(0))

        return dl_dx
        
    def params(self):
        return [(self.w, self.b), (self.dl_dw, self.dl_db)]
    
    def update_params(self, eta):
        self.w = self.w - eta * self.dl_dw
        self.b = self.b - eta * self.dl_db
        
    def reset_gradient(self):
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def reset_params(self):
        # Initialize weights
        xavier_normal_(self.w)
        self.b.normal_(0, self.epsilon)
