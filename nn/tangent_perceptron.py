import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import TangentLin, TangentNonLin

class TangentPerceptron(nn.Module):

    '''
    A perceptron for complex tangent vector features
    The equivalent of a fully-connected linear layer 
    followed by an ReLU
    '''
    
    def __init__(self, in_channels, out_channels):
        super(TangentPerceptron, self).__init__()

        self.lin = TangentLin(in_channels, out_channels)
        self.nonlin = TangentNonLin(out_channels)

    def forward(self, x):
        
        return self.nonlin(self.lin(x))
