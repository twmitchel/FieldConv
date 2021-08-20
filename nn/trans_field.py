import numpy as np

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from utils.field import softAbs, softAngle, softAbsolute, isZero

def weightContribReal(contribAng, contribMag, zonalAng, zonalMag, phase):

    phi = softAngle(torch.sum( contribAng[:, None, ...] * zonalAng[None, ...], dim=3));
    
    rho = softAbsolute(torch.sum( contribMag[:, None, ...] * zonalMag[None, ...], dim=3));
    
    return torch.sum(torch.polar(rho, phi), dim=-1);
    

def weightContribOffset(contribAng, contribMag, zonalAng, zonalMag, phase):

    phi = softAngle(torch.sum(contribAng[:, None, ...] * zonalAng[None, ...], dim=3)) + phase[None, ...]
    
    rho = softAbsolute(torch.sum(contribMag[:, None, ...] * zonalMag[None, ...], dim=3))
    
    return torch.sum(torch.polar(rho, phi), dim=-1)

    
class TransField(nn.Module):

    '''
    Learned 'gradient' operation, taking scalar features to equivariant 
    tangent vector features as discussed in 
    section C of the supplement ( Equations (2) - (3) )
    
    Inputs:
    
    in_channels: # of in channels
    
    out_channels: # of out channels
    
    n_rings: Number of radial bins for filters
    
    ftype: 'Type' of filter, increasing number of parameters for fixed band limit and number of radial bins
    
            0: Radially symmmetric, real-valued filters in L^2(D)
            
            1: Radially symmetric, real-valued filters with per-channel phase offsets
    '''
    
    def __init__(self, in_channels, out_channels, n_rings=6, ftype=1):
        super(TransField, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R = n_rings;
        self.ftype = ftype
        
        if (ftype == 0):  
            self.zonalAng = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, n_rings)); 
            self.zonalMag = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, n_rings));
            self.register_buffer('phase', torch.zeros(out_channels, in_channels));
            
            self.WR = weightContribReal     
            
        else:
            self.zonalAng = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, n_rings)); 
            self.zonalMag = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, n_rings));
            self.phase = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
            
            torch.nn.init.xavier_uniform_(self.phase)
            
            self.WR = weightContribOffset

            
        torch.nn.init.xavier_uniform_(self.zonalAng)
        torch.nn.init.xavier_uniform_(self.zonalMag)


    def forward(self, x, supp_edges, lift_sten):
        
        '''
        Inputs:
        
        x: (N X in_channels) (real - cloat), Input scalar features
        
        supp_edges: (E X 2) (long),  Filter support edges determining which points contribute to convolutions
                    (j, i): j --> i (source to target)
                    
        lift_sten: (E X R X 2), (complex - cfloat), pre-computed 'stencil' (a
                   Radial bin weights + 0th and 1st angular frequencies determined by log_j(i), pre-multiplied by
                   by parallel transport exp( 1i * \varphi_{j --> i}) and integration weight w_j 
        

        Output:
        
        y: (N X out_channels) (complex - cfloat), Output tangent vector features:
        
        '''
        N = x.size()[0]

        ## First perform aggregations for the directional component of our new features
        ## Equation (2) in the supplement
        ## log_i (j) = -1 * P_{i <-- j}( log_j (i) ), parallel transport 'reflects' the logarithm

        xDiff = x[supp_edges[:, 0], ...] - x[supp_edges[:, 1], ...]
        
        contribAng = -1.0 * scatter_add( xDiff[..., None] *  lift_sten[:, None, :, 1], supp_edges[:, 1], dim=0, dim_size=N)
        
        ## Aggregations for magnitude of our new features
        ## Equation (3) in the supplement
        contribMag = scatter_add( x[supp_edges[:, 0], ..., None] * softAbs(lift_sten[:, None, :, 0]), supp_edges[:, 1], dim=0, dim_size=N);
        
        ## Weight, combine components and return tangent vector features
        return self.WR(contribAng, contribMag, self.zonalAng, self.zonalMag, self.phase)


                
