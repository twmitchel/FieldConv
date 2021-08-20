import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.inits import zeros
from utils.field import isOrigin

class TangentNonLin(nn.Module):

    '''
    Applies non-linearities to the radial component of complex features
    Equation (8) in the paper
    Also known as 'modReLU'
    '''
    
    def __init__(self, in_channels):
        super(TangentNonLin, self).__init__()

        self.bias = nn.Parameter(torch.Tensor(1, in_channels));
        
        torch.nn.init.zeros_(self.bias)

            
    def forward(self, x):

        xOut = x.clone();

        nzInd = torch.nonzero(torch.logical_not(isOrigin(x)));
        
        theta = torch.angle(x[nzInd[:, 0], nzInd[:, 1]]);
        r = torch.abs(x[nzInd[:, 0], nzInd[:, 1]]);
        
        xOut[nzInd[:, 0], nzInd[:, 1]] = torch.polar( F.relu(r + self.bias[0, nzInd[:, 1]]), theta);
        
        return xOut;
        
        
        