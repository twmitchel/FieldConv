import numpy as np

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros

from utils.field import complex_prod, softNormalize

class TransField(MessagePassing):

    def __init__(self, in_channels, out_channels, n_rings=3, offset=True):
        super(TransField, self).__init__(aggr='add', flow='target_to_source', node_dim=0)

        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.n_rings = n_rings
        self.offset = offset 
        

        self.radial = Parameter(torch.Tensor(out_channels, in_channels, n_rings, 2));

        if self.offset:
            self.phase = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.register_parameter('phase', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.radial)
        glorot(self.phase)


    def forward(self, x, edge_index, precomp):
        '''
        x: n_nodes x in_channels  f 
        edge_index: 2 x n_edges
        precomp: n_edges x 2 x R (rho) x 2
        connection: n_edges x 2
        
        out: n_nodes x out_channels x 2
        '''
        
        return self.propagate(edge_index=edge_index, x=x, precomp=precomp)


    def message(self, x_j, x_i, precomp):
        """
        :param x_j: n_edges x in_channels 
        :param precomp:  n_edges x 2 (m = 0, 1) x n_rings x 2 
        :return: the message from each target to the source nodes n_edges x in_channels x n_rings x 2 (R, C)
        """
        
        (E, iC), R = x_j.size(), self.n_rings

        # Set up result tensors
        res = torch.cuda.FloatTensor(E, iC, R, 3).fill_(0)       
        
        # Magnitude
        res[..., 0] =  torch.mul(x_j[:, :, None], precomp[:, 0, None, :, 0])

        # Direction
        res[..., 1:] = torch.mul( precomp[:, 1, None, :, :], (x_j[:, :, None, None] - x_i[:, :, None, None]) )
        
        return res;
       


    def update(self, aggr_out):
        """       
        :param aggr_out: the result of the aggregation operation: n_nodes x in_channels x n_rings x 4 or 2

        :radial: out_channels x in_channels x n_rings x 2
        
        :phase: out_channels x in_channels x 2
        
        :return: the new trans. field for x: n_nodes x out_channels x 2
        """
        (E, iC, R, _), oC = aggr_out.size(), self.out_channels
                
        res = torch.cuda.FloatTensor(E, oC, 2).fill_(0);
        
        # Magnitude
        h = torch.sum( torch.mul(aggr_out[:, None, :, :, 0], self.radial[None, :, :, :, 0]), dim=3);
                      
        # Direction                     
        T = torch.sum( torch.mul(aggr_out[:, None, :, :, 1:], self.radial[None, :, :, :, 1, None]), dim=3);
                      
        hT = torch.mul(h[..., None], softNormalize(T));
            
        if self.offset:
                
            exp = torch.cat( (torch.cos(self.phase[..., None]), torch.sin(self.phase[..., None])), dim=2)
               
            hT = complex_prod(hT.clone(), exp[None, ...])

        
        return torch.sum(hT, dim=2)

                
