import os.path as osp
from math import pi as PI

import torch

from torch_scatter import scatter_add
from torch_geometric.utils import degree

from utils.field import root_weights 

class FCPrecomp(object):
    r"""Precomputation for Harmonic Surface Networks.
    Asserts that a logmap and vertex weights have been computed
    and stored in data.edge_attr, data.weight.

    .. math::
        w_j \mu_{\q}(r_{ij}) e^{\i m\theta_{ij}}\right

    Args:
        n_rings (int, optional): number of rings used to parametrize
            the radial profile, defaults to 2.
        max_order (int, optional): the maximum rotation order of the network,
            defaults to 1.
        max_r (float, optional): the radius of the kernel,
            if not supplied, maximum radius is used.
        cache_file (string, optional): if set, cache the precomputation
            in the given file and reuse for every following shape.
    """

    def __init__(self, n_conv_rings=2, n_corr_rings=2, band_limit=1, max_r=None):
        self.n_conv_rings = n_conv_rings
        self.n_corr_rings = n_corr_rings
        self.band_limit = band_limit
        self.max_r = max_r


    def __call__(self, data):
        assert hasattr(data, 'edge_attr')
        assert hasattr(data, 'weight')


        # Conveniently name variables
        (row, col), pseudo, weight = data.edge_index, data.edge_attr, data.weight
        N, M, R_conv, R_corr = row.size(0), self.band_limit, self.n_conv_rings, self.n_corr_rings
        r, theta = pseudo[:, 0], pseudo[:, 1]

        # Normalize radius to range [0, 1]
        r =  r / self.max_r if self.max_r is not None else r / r.max()

        # Compute interpolation weights for the radial profile function
        scatter_weights = root_weights(r, R_conv);
        gather_weights = root_weights(r, R_corr);

            
        # Compute exponential component for each point
        angles = theta.view(-1, 1) * torch.arange(M+1).float().view(1, -1).to(theta.device) * 2 * PI
        exponential = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) # [N, M + 1, 2]
        
        exp_corr = exponential[:, :2, :];


        # Finally, normalize weighting for every neighborhood and append to log
        w_gather = weight[data.edge_index[1], 0];
        w_scatter = weight[data.edge_index[0], 0];
        
        
        w_gather = w_gather / (1e-12 + scatter_add(w_gather, row)[row]);
        
        w_scatter = w_scatter / (1e-12 + scatter_add(w_scatter, data.edge_index[1, :].squeeze())[data.edge_index[1, :].squeeze()])
       

        # Combine precomputation components
        
        pcmp_scatter = w_scatter.view(N, 1, 1, 1)* scatter_weights.view(N, 1, R_conv, 1) * exponential.view(N, M+1, 1, 2)
        
        pcmp_gather = w_gather.view(N, 1, 1, 1) * gather_weights.view(N, 1, R_corr, 1) * exp_corr.view(N, 2, 1, 2)
        
        logNrm = torch.mul(exponential[:, 1, :], r[:, None]);
        

        data.pcmp_scatter = pcmp_scatter.contiguous() # [N, M+1, R, 2]
        data.pcmp_gather = pcmp_gather.contiguous()
        data.pcmp_echo = torch.cat( (logNrm, w_scatter[:, None]), dim=1 ) # N x 2
        

        return data

    def __repr__(self):
        return '{}(n_conv_rings={}, n_corr_rings={}, band_limit={}, max_r={})'.format(self.__class__.__name__,
                                                  self.n_conv_rings, self.n_corr_rings, self.band_limit, self.max_r)