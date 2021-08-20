import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import FieldConv, TangentLin, TangentNonLin

class FCResNetBlock(torch.nn.Module):
    '''
    The FCResNet block from section 5 of the paper (Figure 2).
    
    Two field convoutions followed by non-linearities with a residual connection
    
    The basic, modularized building blocks for incorperating field convolutions
    in surface networks. In other words, we reccomennd trying this first. 
    
    
    Inputs:
    
    in_channels: # of in channels
    
    out_channels: # of out channels
    
    band_limit: Filter band limit (number of angular frequencies considered in Fourier decomposition)

    n_rings: Number of radial bins for filters
    
    ftype: 'Type' of filter, increasing number of parameters for fixed band limit and number of radial bins
    
            0: Real-valued filters in L^2(D)
            
            1: Real-valued filters with per-channel phase offsets
            
            2: Complex-valued filters in L^2(D) (offsets are subsumed by complex values)
            
    frontload: Which convolution block maps in_channels to out_channels
            
            False (default): conv1: in_channels --> out_channels, conv2: out_channels --> out_channels
            
            True: conv1: in_channels --> in_channels, conv2: in_channels --> out_channels
    
    '''
    
    def __init__(self, in_channels, out_channels, band_limit=1, n_rings=6, ftype=1, frontload=False):
        super(FCResNetBlock, self).__init__()
        
        iC1 = in_channels
        oC2 = out_channels;        
        
        if (frontload == False):
            oC1 = out_channels;
            iC2 = out_channels;
        else:
            oC1 = in_channels;
            iC2 = in_channels;
            
        self.conv1 = FieldConv(iC1, oC1, band_limit=band_limit, n_rings=n_rings, ftype=ftype)
        
        self.conv2 = FieldConv(iC2, oC2, band_limit=band_limit, n_rings=n_rings, ftype=ftype) 

        self.nonlin1 = TangentNonLin(oC1)
        self.nonlin2 = TangentNonLin(oC2)
        self.res = TangentLin(iC1, oC2);
    

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
        
        y: (N X out_channels) (complex - cfloat), Output tangent vector features
        '''
         
        x_conv = self.nonlin1( self.conv1(x, supp_edges, supp_sten) );
                                                 
        x_conv = self.conv2( x_conv, supp_edges, supp_sten)
        
        return self.nonlin2(self.res(x) + x_conv);
        
     
    
  