from model import Module
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

        # Initialize velocities
        self.uw = torch.zeros(self.w.size())
        self.ub = torch.zeros(self.b.size())

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

    def parameters(self):
        return (self.w, self.b)

    def SGD(self, eta, momentum):
        self.uw = self.uw * momentum + self.dl_dw * eta
        self.ub = self.ub * momentum + self.dl_db * eta

        self.w = self.w - self.uw
        self.b = self.b - self.ub

    def zero_gradient(self):
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def reset_parameters(self):
        # Initialize weights
        xavier_normal_(self.w)
        self.b.normal_(0, self.epsilon)
        self.uw.zero_()
        self.ub.zero_()
