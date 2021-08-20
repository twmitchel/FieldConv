from math import pi as PI
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from utils.field import softSqrt, softAngle, isOrigin, softAbs

rt2 = np.sqrt(2.0);

## Binning -- rasterize the disk
def diskMap(n_bins):
    
    w = 2 * n_bins + 1
    dim = 0;
    ind = []
    for i in range(0, w):
        for j in range(0, w):
            if  (i - n_bins)*(i - n_bins) + (j - n_bins) * (j - n_bins) <= (n_bins + 0.25) * (n_bins + 0.25):
                ind.append(w * i + j);
                dim = dim + 1;
    
    ind = torch.Tensor(ind).long()
    dMap = torch.empty(w*w).long().fill_(0)
    
    dMap[ind] = torch.arange(ind.size()[0])
    
    return dMap, dim


def rasterize(p, dMap, n_bins):
    
    w = 2 * n_bins + 1;
    
    rast = torch.empty(p.size()[0], 4, device=p.device).float().fill_(0)
    ind = torch.empty(p.size()[0], 4, device=p.device).long().fill_(0)
    
    rng = torch.arange(p.size()[0], device=p.device)
    
    p = torch.view_as_real(torch.mul(p, n_bins));
    
    pC = torch.clamp(torch.ceil(p).long(), -n_bins, n_bins).long()
    pF = torch.clamp(torch.floor(p).long(), -n_bins, n_bins).long()
    
    # Store bilinear interpolation weights + bin indices per edge
    rast[rng, 0] = torch.prod(pC - p, dim=1)
    
    ind[rng, 0] = dMap[w*(pF[..., 0] + n_bins) + (pF[..., 1] + n_bins)]
    
    rast[rng, 1] = torch.prod(p - pF, dim=1)
    
    ind[rng, 1] = dMap[w*(pC[..., 0] + n_bins) + (pC[..., 1] + n_bins)]
    
    rast[rng, 2] = (p[..., 0] - pF[..., 0]) * (pC[..., 1] - p[..., 1])
    
    ind[rng, 2] = dMap[w*(pC[..., 0] + n_bins) + (pF[..., 1] + n_bins)]
    
    rast[rng, 3] = (pC[..., 0] - p[..., 0]) * (p[..., 1] - pF[..., 1])
    
    ind[rng, 3] = dMap[w*(pF[..., 0] + n_bins) + (pC[..., 1] + n_bins)]
    
    return rast, ind



class ECHO(nn.Module):
    '''
    ECHO Descriptors
    
    Given a surface vector field, computes ECHO descriptors from the paper "ECHO: Extended Convolution Histogram 
    of Orientations for Local Surface Description" for each input channel.
    
    Inputs:
    
    channels: # of channels
        
    n_bins: Number of bins per unit radius, descriptor resolution
            is approximately PI *(n_bins + 0.5) * (n_bins + 0.5)
    '''
    
    def __init__(self, channels, n_bins=2):
        super(ECHO, self).__init__()

        self.channels = channels
        
        self.n_bins = n_bins;
        
        dMap, dim = diskMap(n_bins);
        
        self.register_buffer('dMap', dMap);
        
        self.hdim = dim
                             

    def forward(self, x, supp_edges, ln,  wxp):
        
        '''
        Inputs:
        
        x: (N X channels) (complex - cfloat), Input tangent vector features 
        
        supp_edges: (E X 2) (long),  Filter support edges determining which points contribute to convolutions
                    (j, i): j --> i (source to target)
         
        ln: (E) (complex - cfloat), log_j(i) as a complex number
                                      
        wxp: E (complex - cfloat), Transport pre-multiplied by integration weights
        
        Output:
        
        y: (N X channels x n_des * n_des) (complex - cfloat), Output per-channel n-dimensional ECHO descriptors
        '''
        N, E, C, nB, dS = x.size()[0], supp_edges.size()[0], self.channels, self.n_bins, self.hdim

        
        ## Zero-valued features don't contribute to descriptors, so we're going to
        ## perform some gymnastics to make sure we're only aggregating edges transporting
        ## non-zero features.
                
        spatial_edges = supp_edges.repeat(C, 1);
        cInd = torch.arange(C, device=x.device).repeat_interleave(E);
        edge_list = torch.arange(E, device=x.device).repeat(C);
        
        nzInd = torch.nonzero( torch.logical_not(isOrigin(x))[spatial_edges[:, 0], cInd] ).squeeze(-1)
        
        spatial_edges = spatial_edges[nzInd, :];
        cInd = cInd[nzInd]
        edge_list = edge_list[nzInd];
        
        ## Align local coordinate systems w.r.t. to frames defined by features
        aligned = ln[edge_list] * torch.conj(torch.polar(torch.ones_like(x.real), softAngle(x)))[spatial_edges[:, 0], cInd]
        
        # Compute target disk rasterization
        rast, ind = rasterize(aligned, self.dMap, nB)
        
        ## Multiply features by integration weights + transport
        xW = (x[spatial_edges[:, 0], cInd] * wxp[edge_list])
        
        ## Target indices
        target = C*dS*spatial_edges[:, 1] + dS*cInd
        
        # Compute descriptor by scattering votes

        return softAbs( torch.reshape( 
                        scatter_add( xW * rast[:, 0], target + ind[:, 0], dim=0, dim_size=N*C*dS) +
                        scatter_add( xW * rast[:, 1], target + ind[:, 1], dim=0, dim_size=N*C*dS) +
                        scatter_add( xW * rast[:, 2], target + ind[:, 2], dim=0, dim_size=N*C*dS) +
                        scatter_add( xW * rast[:, 3], target + ind[:, 3], dim=0, dim_size=N*C*dS),
                        (N, C, dS) ) )
        
    


