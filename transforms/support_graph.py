import os.path as osp
import torch
import numpy as np
import vectorheat as vh
import trimesh as tm

from torch_sparse import coalesce
from torch_geometric.nn import radius, fps
from torch_geometric.utils import to_undirected


class SupportGraph(object):
    r"""Creates a radius graph for multiple pooling levels.
    The nodes and adjacency matrix for each pooling level can be accessed by masking
    tensors with values for nodes and edges with data.node_mask and data.edge_mask, respectively.

    Edges can belong to multiple levels,
    therefore we store the membership of an edge for a certain level with a bitmask:
        - The bit at position 2 * n corresponds to the edges used for pooling to level n
        - The bit at position 2 * n + 1 corresponds to the edges used for convolution in level n

    To find out if an edge belongs to a level, use a bitwise AND:
        `edge_mask & (0b1 << lvl) > 0`

    Args:
        ratios (list): the ratios for downsampling at each pooling layer.
        radii (list): the radius of the kernel support for each scale.
        max_neighbours (int, optional): the maximum number of neighbors per vertex,
            important to set higher than the expected number of neighbors.
        sample_n (int, optional): if provided, constructs a graph for only sample_n vertices.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cache_file (string, optional): if set, cache the precomputation
            in the given file and reuse for every following shape.
    """

    def __init__(self, epsilon, sample_n=None, flow='source_to_target'):

        self.epsilon = epsilon
        self.sample_n = sample_n
        self.flow = flow


    def __call__(self, data):

        data.edge_attr = None
        pos = data.pos
        
        pos_tm = pos.cpu().numpy()
        faces_tm = data.face.cpu().numpy().T

        
        # Sample points on the surface using farthest point sampling if sample_n is given
        # We use a custom biharmonic FPS sampling regime 
     
        if hasattr(data, 'sample_idx'):
            sample_idx = data.sample_idx
        else:
            if self.sample_n is not None and not self.sample_n > data.pos.size(0):
                sample_idx = fps(pos, batch=None, ratio=self.sample_n / data.pos.size(0)).sort()[0]
            else:
                sample_idx = torch.arange(data.num_nodes)
           
            data.sample_idx = sample_idx

        original_idx = torch.arange(sample_idx.size(0))
        pos = pos[sample_idx]
        
        ## Convolution edges
        radius_edges = radius(pos, pos, self.epsilon, batch_x=None, batch_y=None, max_num_neighbors=512)
        edge_index = original_idx[radius_edges]

        data.edge_index = edge_index;

        return data

    def __repr__(self):
        return '{}(epsilon={}, sample_n={})'.format(self.__class__.__name__, self.epsilon, self.sample_n)
