import os.path as osp
import torch
import numpy as np
import fcutils as fc
from math import pi as PI
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import degree

class computeLogXPort(object):
    '''
    Uses the Vector Heat Method (Sharp et al. 2019) to precompute parallel transport,
    logarithmic map, and integration weights for each edge.

    Adapted from: https://github.com/rubenwiersma/hsn
    '''

    def __init__(self):
        []

    def __call__(self, data):

        # Prepare data for vector heat method
        pos, face, supp_edges= data.pos.cpu().numpy(), data.face.cpu().numpy().T, data.supp_edges.cpu().numpy()
        sample_idx = data.sample_idx.cpu().numpy()
        deg = degree(data.supp_edges[:, 0]).cpu().numpy()
        
        # Compute parallel transport,
        # logarithmic map and vertex lumped mass matrix for each edge..
  
        result = fc.precompute(pos, face, supp_edges, deg, sample_idx)
        weights = fc.weights(pos, face, sample_idx, np.arange(len(sample_idx)))
        
        # Transport
        # (j, i), j --> i (target to source)
        data.xp = torch.view_as_complex(torch.from_numpy(result[:, :2]).float().contiguous())

        # Integration weights
        data.w = torch.from_numpy(weights).float();

        # Logarithm
        # Source to target: edges (j, i) j-->i
        coords = torch.view_as_complex(torch.from_numpy(result[:, 2:]).float().contiguous())
        
        # Compute polar coordinates from cartesian coordinates
        r = torch.abs(coords)
        theta = torch.angle(coords)
        
        data.logMag = r;
        data.logAng = theta;

        
        return data


    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
