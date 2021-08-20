import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import ECHO, FieldConv, TangentPerceptron, TangentLin, TangentNonLin
from utils.field import softAbs


## Computes the resolution of the ECHO descriptor
def histDim (n_bins):
    
    w = 2 * n_bins + 1
    dim = 0;
    for i in range(0, w):
        for j in range(0, w):
            if  (i - n_bins)*(i - n_bins) + (j - n_bins) * (j - n_bins) <= (n_bins + 0.25) * (n_bins + 0.25):
                dim = dim + 1;
    return dim

class ECHOBlock(torch.nn.Module):
    '''
    The ECHO Block as described in section 5 of the paper. 
    Converts tangent vector features into scalar feature descriptors
    
    Inputs:
    
    in_channels: # of in channels
    
    out_channels: # of out channels 
    
    n_des: # of descriptors to compute (default: in_channels)
    
    n_bins: Number of bins per unit radius, descriptor resolution
            is approximately PI *(n_bins + 0.5) * (n_bins + 0.5)

    band_limit: Filter band limit (number of angular frequencies considered in Fourier decomposition)

    n_rings: Number of radial bins for filters
    
    ftype: 'Type' of filter, increasing number of parameters for fixed band limit and number of radial bins
    
            0: Real-valued filters in L^2(D)
            
            1: Real-valued filters with per-channel phase offsets
            
            2: Complex-valued filters in L^2(D) (offsets are subsumed by complex values)
    '''
    
    def __init__(self, in_channels, out_channels, n_des=None, n_bins=3, band_limit=1, n_rings=6, ftype=1):
        super(ECHOBlock, self).__init__()
        
        if n_des is None:
            n_des = in_channels     
                
        self.conv = FieldConv(in_channels, n_des, band_limit, n_rings, ftype);
        
        self.nonlin = TangentNonLin(in_channels)
                   
        self.echo = ECHO(n_des, n_bins);
        
        mid_channels = n_des * histDim(n_bins)

        # MLP
        self.lin1 = nn.Linear(mid_channels, 128)
        self.lin2 = nn.Linear(128, 64);
        self.lin3 = nn.Linear(64, out_channels)

        # Residual connection
        self.res = nn.Linear(in_channels, out_channels)
        
        
                
    def forward(self, x, supp_edges, supp_sten, ln, wxp):
        
        '''
        Inputs:
        
        x: (N X channels) (complex - cfloat), Input tangent vector features 
        
        supp_edges: (E X 2) (long),  Filter support edges determining which points contribute to convolutions
                    (j, i): j --> i (source to target)
                    
        
        supp_sten: ( E X R X (2*B+1) ) (complex - cfloat), pre-computed convolution 'stencil' 
                   Radial bin weights + angular frequencies determined by log_j(i), pre-multiplied by
                   by parallel transport exp( 1i * \varphi_{j --> i}) and integration weight w_j 
        
        ln: (E) (complex - cfloat), log_j(i) as a complex number
                           
        wxp: (E) (complex - cfloat), Transport pre-multiplied by integration weights
        '''
                
        xE = self.nonlin(self.conv(x, supp_edges, supp_sten));
                
        xE = self.echo(xE, supp_edges, ln, wxp);
        
        xE = torch.reshape(xE, (xE.size(0), -1))
            
        xE = F.relu(self.lin1(xE))

        xE = F.relu(self.lin2(xE))

        return self.lin3(xE) + self.res(softAbs(x))
        

