import os.path as osp
from math import pi as PI

import torch

from torch_scatter import scatter_add
from torch_geometric.utils import degree

## Interpolation weights, sampled on a square root scale (equi-area bins)
def radialInterpolant(r, n_rings):
    
    samples = torch.sqrt(torch.div(torch.arange(n_rings, device=r.device), n_rings-1))
    
    diff = torch.sub(samples[None, :], r[:, None])
    diff[diff < 0] = 1e8;

    cIndex = torch.min(diff, dim=1).indices;
    cIndex[cIndex==0] = 1;
    fIndex = torch.sub(cIndex, 1);
    
    weights = torch.empty(r.size()[0], n_rings, device=r.device).float().fill_(0)
    rng = torch.arange(r.size()[0], device=r.device)
    
    weights[rng, cIndex] = (r - samples[fIndex]) / (samples[cIndex] - samples[fIndex])
    weights[rng, fIndex] = torch.ones_like(rng) - weights[rng, cIndex]

    return weights


class FCPrecomp(object):
    '''
    Organizes edge data for field convolutions
    
    More specifically, it precomputes and pre-mulitplies the radial bin weights (rotations preserve distances),
    angular frequencies of the log angles, transport, and integration weights to facilitate filter 
    evalutions (Equations (6)-(7) in the paper)
        
    Inputs:
    
    band_limit: Filter band limit (number of angular frequencies considered in Fourier decomposition)

    n_rings: Number of radial bins for filters

    epsilon: Radius of filter support 
    '''

    def __init__(self, band_limit, n_rings, epsilon):
        self.B = band_limit
        self.R = n_rings
        self.max_r = epsilon
        
       
    def __call__(self, data):
        
        # Debug
        #assert hasattr(data, 'logMag')
        #assert hasattr(data, 'logAng')
        #assert hasattr(data, 'w')
        

        r, theta, w, supp_edges, xp = data.logMag, data.logAng, data.w, data.supp_edges, data.xp

        B, R = self.B, self.R

        
        # Normalize radius to range [0, 1], remove edges outside of support
        r =  r / self.max_r
        
        validInd = torch.nonzero(r <= 1.0).squeeze(-1);
        
        r = r[validInd]
        theta = theta[validInd]
        supp_edges = supp_edges[validInd, :]
        xp = xp[validInd]
        
        ## Represent log as complex number
        ln = torch.polar(r, theta);
        
        # Compute radial interpolation stencil (E X R)
        rSten = radialInterpolant(r, R);
        
        ## Compute angular frequencies of the log map's polar angles (E X (2 * B + 1) )
        freq = torch.arange(-B, B+1, device=theta.device)[None, ...] * theta[:, None];
        fSten = torch.polar(torch.ones_like(freq), freq);
        
        # Normalize weights (optional)
        w_scatter = w[supp_edges[:, 0], 0] / (1e-12 + scatter_add(w[supp_edges[:, 0], 0], supp_edges[:, 1])[supp_edges[:, 1]])
        #w_scatter = w[supp_edges[:, 0], 0];
        
        
        # Multiply bin weights by weights + transport
        wxp = w_scatter *  xp;
        
        # Construct convolution stencil (E X R X (2 * B + 1) )
        supp_sten = rSten[..., None] * fSten[:, None, :] * wxp[:, None, None]
        
        return supp_edges, supp_sten, ln, wxp

    def __repr__(self):
        return '{}(n_rings={}, epsilon={})'.format(self.__class__.__name__,
                                                  self.R, self.max_r)