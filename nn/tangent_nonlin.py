import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.inits import zeros
from utils.field import softNorm, killNormalize

class TangentNonLin(nn.Module):

    def __init__(self, in_channels, fnc=F.relu):
        super(TangentNonLin, self).__init__()

        self.fnc = fnc
        self.bias = nn.Parameter(torch.Tensor(in_channels));
        zeros(self.bias)

            


    def forward(self, x):
           
        h = softNorm(x);

        T = killNormalize(x);

        c = self.fnc(h+ self.bias[None, :])

        return torch.mul(T, c[..., None])    
    
