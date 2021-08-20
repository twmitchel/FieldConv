import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
from torch_scatter import scatter_add, scatter_min
from torch_geometric.data import Data

EPS = 1e-7

def isZero(x, eps=EPS):
    
    return torch.logical_and( torch.lt(x, eps), torch.gt(x, -eps) );

def isOrigin(z, eps=EPS):
    
    return torch.logical_and( isZero(z.real, eps), isZero(z.imag, eps) );

def softAbsolute(x):
    
    xOut = x;
    
    negInd = torch.nonzero(x < 0, as_tuple=True);
    
    xOut[negInd] = -1.0*x[negInd];
    
    return xOut;


def softAbs(z, eps=EPS):
    
    m = torch.zeros_like(z.real)
    
    nzInd = torch.nonzero(torch.logical_not(isOrigin(z, eps)), as_tuple=True);
    
    m[nzInd] = torch.abs(z[nzInd])
        
    return m;


def softAngle(z, eps=EPS):
    
    t = torch.zeros_like(z.real)
    
    nzInd = torch.nonzero(torch.logical_not(isOrigin(z, eps)), as_tuple=True);
    
    t[nzInd] = torch.angle(z[nzInd])
        
    return t;
 

def softSqrt(x, eps=EPS):
    
    xOut = torch.zeros_like(x);
    nzInd = torch.nonzero(torch.logical_not(isZero(x, eps)), as_tuple = True);
    
    xOut[nzInd] = torch.sqrt(x[nzInd]);
    
    return xOut;



