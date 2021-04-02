import os.path as osp
import torch
import numpy as np
import vectorheat as vh
from math import pi as PI
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import degree

class VectorHeat(object):
    r"""Uses the Vector Heat Method to precompute parallel transport,
    logarithmic map, and integration weight for each edge.

    The data object should hold positions and faces,
    as well as edge indices denoting local regions.

    Args:
        cache_file (string, optional): if set, cache the precomputation
            in the given file and reuse for every following shape.
    """

    def __init__(self):
        ;

    def __call__(self, data):
        assert hasattr(data, 'pos')
        assert hasattr(data, 'face')
        assert hasattr(data, 'edge_index')

        # Prepare data for vector heat method
        pos, face, edge_index = data.pos.cpu().numpy(), data.face.cpu().numpy().T, data.edge_index.cpu().numpy().T
        sample_idx = data.sample_idx.cpu().numpy()
        deg = degree(data.edge_index[0]).cpu().numpy()
        
        # Use Vector Heat method to compute parallel transport,
        # logarithmic map and vertex lumped mass matrix for each edge.
        # We provide the degree of each vertex to the vector heat method,
        # so it can easily iterate over the edge tensor.
  
        vh_result = vh.precompute(pos, face, edge_index, deg, sample_idx)
        vh_weights = vh.weights(pos, face, sample_idx, np.arange(len(sample_idx)))

        # Unpack results and create torch tensors.
        data.connection = torch.from_numpy(vh_result[:, :2]).float()
        
        # Computed as (j, i), j --> i
        # Invert so   (j, i), j <-- i (target to source)

        data.connection[:, 1] = -data.connection[:, 1]
        data.connection = data.connection.contiguous()
                
        data.weight = torch.from_numpy(vh_weights).float()
        

        # Logarithm
        # Source to target: edges (j, i) j-->i
        
        coord = torch.from_numpy(vh_result[:, 2:]).float()
        
        # Compute polar coordinates from cartesian coordinates
        r = coord.norm(dim=1)
        theta = torch.atan2(coord[:, 1], coord[:, 0])
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)
        theta = theta / (2 * PI)
        data.edge_attr = torch.stack((r, theta), dim=-1)
        
        return data


    def __repr__(self):
        return '{}'.format(self.__class__.__name__)