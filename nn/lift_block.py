import torch
import torch.nn as nn

from nn import  TransField, TangentNonLin

class LiftBlock(torch.nn.Module):
    '''
    Lifts input scalar features to tangent vector features 
    via the learned 'gradient' operation discussed in section B
    of the supplement, followed by a non-linearity
    
    Inputs:
    
    in_channels: # of in channels
    
    out_channels: # of out channels
    
    n_rings: Number of radial bins for filters
    
    ftype: 'Type' of filter, increasing number of parameters for fixed band limit and number of radial bins
    
            0: Radially symmmetric, real-valued filters in L^2(D)
            
            1: Radially symmetric, real-valued filters with per-channel phase offsets
    '''
    
    def __init__(self, in_channels, out_channels, n_rings=6, ftype=1):
        super(LiftBlock, self).__init__()
        
        
        self.field = TransField(in_channels, out_channels, n_rings=n_rings, ftype=ftype)
        
        self.nonlin = TangentNonLin(out_channels);

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
        
        y: (N X out_channels) (complex - cfloat), Output tangent vector features):
        
        '''
            
        return self.nonlin(self.field(x, supp_edges, lift_sten))
        