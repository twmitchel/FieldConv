import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add

from utils.field import softAngle, isOrigin

def weightContribReal(contrib, zonal, spherical, phase, B):
        
    coeff = torch.cat(( torch.conj(torch.view_as_complex(spherical)), zonal[..., None], torch.view_as_complex(spherical)), dim=3);
    
    return torch.div( torch.sum( contrib[:, None, ...] * coeff[None, ...], dim=(2, 3, 4)), 2*B+1);
    
def weightContribOffset(contrib, zonal, spherical, phase, B):
    
    coeff = torch.cat(( torch.conj(torch.view_as_complex(spherical)), zonal[..., None], torch.view_as_complex(spherical)), dim=3);
    

    weighted = torch.sum( contrib[:, None, ...] * coeff[None, ...], dim=3)
    
    phases = torch.cat( (torch.flip(phase[:, :, 1:], [2]), phase), dim=-1)
    
    return torch.div( torch.sum( weighted * torch.polar(torch.ones_like(phases), phases)[None, ...], dim=(2, 3)), 2*B + 1);


def weightContribComplex(contrib, zonal, spherical, phase, B):
    
    
    coeff = torch.cat(( torch.view_as_complex(spherical)[..., :B], torch.view_as_complex(zonal)[..., None], torch.view_as_complex(spherical)[..., B:]), dim=3);
    
    return torch.div( torch.sum( contrib[:, None, ...] * coeff[None, ...], dim=(2, 3, 4)), 2*B+1);
                        
    
class FieldConv(nn.Module):
    
    '''
    Field Convolution
    
    As described in section 4 of the paper ( Equations (4) and (7) )
    
    Inputs:
    
    in_channels: # of in channels
    
    out_channels: # of out channels
    
    band_limit: Filter band limit (number of angular frequencies considered in Fourier decomposition)

    n_rings: Number of radial bins for filters
    
    ftype: 'Type' of filter, increasing number of parameters for fixed band limit and number of radial bins
    
            0: Real-valued filters in L^2(D)
            
            1: Real-valued filters with per-channel phase offsets
            
            2: Complex-valued filters in L^2(D) (offsets are subsumed by complex values)
    '''

    def __init__(self, in_channels, out_channels, band_limit=1, n_rings=6, ftype=1):
        super(FieldConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.R = n_rings;
        self.B = band_limit;
        self.ftype = ftype
        
        if (ftype == 0): 
            
            self.zonal = Parameter(torch.Tensor(out_channels, in_channels, n_rings)); 
            self.spherical = Parameter(torch.Tensor(out_channels, in_channels, n_rings, band_limit, 2))
            self.register_buffer('phase', torch.zeros(out_channels, in_channels, band_limit+1));
            
            self.WR = weightContribReal     
            
        elif (ftype==1):
            
            self.zonal = Parameter(torch.Tensor(out_channels, in_channels, n_rings));
            self.spherical = Parameter(torch.Tensor(out_channels, in_channels, n_rings, band_limit, 2))
            self.phase = Parameter(torch.Tensor(out_channels, in_channels, band_limit + 1))
            
            torch.nn.init.xavier_uniform_(self.phase)
            
            self.WR = weightContribOffset
            
        else:
            
            self.zonal = Parameter(torch.Tensor(out_channels, in_channels, n_rings, 2));
            self.spherical = Parameter(torch.Tensor(out_channels, in_channels, n_rings, 2*band_limit, 2))
            self.register_buffer('phase', torch.zeros(out_channels, in_channels, band_limit+1));
            
            self.WR = weightContribComplex
            
        torch.nn.init.xavier_uniform_(self.zonal)
        torch.nn.init.xavier_uniform_(self.spherical)
        




    def forward(self, x, supp_edges, supp_sten):
        
        '''
        Inputs:
        
        x: (N X in_channels) (complex - cfloat), Input tangent vector features 
        
        supp_edges: (E X 2) (long),  Filter support edges determining which points contribute to convolutions
                    (j, i): j --> i (source to target)
                    
        supp_sten: ( E X R X (2*B+1) ) (complex - cfloat), pre-computed convolution 'stencil' 
                   Radial bin weights + angular frequencies determined by log_j(i), pre-multiplied by
                   by parallel transport exp( 1i * \varphi_{j --> i}) and integration weight w_j 
                    
        Output:
        
        y: (N X out_channels) (complex - cfloat), The response, output tangent vector features
        '''
        
        N, B = x.size()[0], self.B

        
        ## Compute feature direction, associated frequencies, and pre-multiply by features
       
        phi = softAngle(x);
        freq = -1.0*torch.arange(-B, B+1, device=phi.device)[None, None, ...] * phi[..., None];
        T = (x[..., None] * torch.polar(torch.ones_like(freq), freq))[supp_edges[:, 0], :, None, :] * supp_sten[:, None, ...];
        
       
        ## All that is left is to multiply by pre-computed stencil, sum contributions...
        contrib = scatter_add(T, supp_edges[:, 1], dim=0, dim_size=N);
        
        ## And weight, reduce and return
        return self.WR(contrib, self.zonal, self.spherical, self.phase, B);
