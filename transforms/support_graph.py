import os.path as osp
import torch
import numpy as np
import fcutils as fc

from torch_sparse import coalesce
from torch_geometric.nn import radius, fps
from torch_geometric.utils import to_undirected


class SupportGraph(object):
    '''
    Computes edges for filter support
    
    Inputs:
    
    epsilon: radius of filter support
    
    sample_n: number of sample points
    
    Adapted from: https://github.com/rubenwiersma/hsn
    '''

    def __init__(self, epsilon, sample_n=None):

        self.epsilon = epsilon
        self.sample_n = sample_n


    def __call__(self, data):
        pos = data.pos
        
        pos_tm = pos.cpu().numpy()
        faces_tm = data.face.cpu().numpy().T

        
        # Sample points on the surface using farthest point sampling if sample_n is given
        # Here we use PyTorch Geometric's built in module for convienence. 
        # In the paper we use custom FPS and radius support modules 
        # based on the biharmonic distance 
     
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

        data.supp_edges = torch.transpose(edge_index, 0, 1);

        return data

    def __repr__(self):
        return '{}(epsilon={}, sample_n={})'.format(self.__class__.__name__, self.epsilon, self.sample_n)
