import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import TangentLin, TangentNonLin

class TangentPerceptron(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):
        super(TangentPerceptron, self).__init__()

        self.lin = TangentLin(in_channels, out_channels, bias)
        self.nonlin = TangentNonLin(out_channels)

    def forward(self, x):
        
        return self.nonlin(self.lin(x))
