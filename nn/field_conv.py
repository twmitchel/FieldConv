import numpy as np

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros


from utils.field import complex_prod, complex_prod_conj, conj, norm2D, softNormalize, softAtan2

class FieldConv(MessagePassing):
    r"""
    Harmonic Convolution from Harmonic Surface Networks

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        prev_order (int, optional): the maximum rotation order of the previous layer,
            should be set to 0 if this is the first layer (default: :obj:`1`)
        max_order (int, optionatl): the maximum rotation order of this convolution
            will convolve with every rotation order up to and including `max_order`,
            (default: :obj:`1`)
        n_rings (int, optional): the number of rings in the radial profile (default: :obj:`2`)
        offset (bool, optional): if set to :obj:`False`, does not learn an offset parameter,
            this is practical in the last layer of a network (default: :obj:`True`)
        separate_streams (bool, optional): if set to :obj:`True`, learns a radial profile
            for each convolution connecting streams, instead of only an m=0 and m=1 convolution
            (default: :obj:`True`)
    """
    def __init__(self, in_channels, out_channels, band_limit = 1, n_rings=6, offset=True):
        super(FieldConv, self).__init__(aggr='add', flow='source_to_target', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_rings = n_rings;
        self.B = band_limit;
        self.offset = offset;
        
        
        self.radial = Parameter(torch.Tensor(out_channels, in_channels, band_limit, n_rings, 2))
        self.real = Parameter(torch.Tensor(out_channels, in_channels, n_rings));
          
        
        if self.offset:
            self.phase = Parameter(torch.Tensor(out_channels, in_channels, band_limit+1))
        else:
            self.register_parameter('phase', None)
            

        self.reset_parameters()
        

    def reset_parameters(self):
        glorot(self.radial)
        glorot(self.real)
        glorot(self.phase)


    def forward(self, x, edge_index, precomp, connection):
        
        '''
        x: n_nodes x in_channels x 2 ({h, T})
        edge_index: 2 x n_edges
        precomp: n_edges x (B + 1) x R (rho) x 2
        connection: n_edges x 2
        grad_index: 2 x g_edges
        edge_grad = g_edges x 2
        
        
        out: n_nodes x out_channels x 3
        '''
        (N, iC, _), B = x.size(), self.B
                
        ## Compute decomposition of signal and transformation field        
        x0 = torch.cuda.FloatTensor(N, iC, 2*B + 1, 2).fill_(0)

        # n_nodes x in_channels x 2

        T = softNormalize(x);

        # m = 0
        x0[:, :, 0, :] = x;
        assert torch.isnan(T).any() == False
        assert torch.isnan(x0).any() == False
        
        if B > 0:
        
            # m = +/- 1
            x0[:, :, 1, :] = complex_prod_conj( x0[:, :, 0, :].clone(), T );
            
            x0[:, :, B+1, :] = complex_prod( x0[:, :, 0, :].clone(), T);
            
            # m = +/- 2, ... B
            for m in range(2, B+1):
                x0[:, :, m, :] = complex_prod_conj( x0[:, :, m-1, :].clone(), T );
            
                x0[:, :, B+m, :] = complex_prod(x0[:, :, B+m-1, :].clone(), T);
                  
        
        '''
        # Tying myself in a knot to avoid a for loop 
        # n_nodes x in_channels x B
        theta = torch.mul(softAtan2(T[..., 1], T[..., 0])[..., None], torch.arange(B)[None, None, :].to('cuda'))
        
        #n_nodes x in_channels x B x 2
        Tm = complex_prod(T[:, :, None, :], torch.cat( (torch.cos(theta[..., None]), torch.sin(theta[..., None])), dim=3) );
        
        x0[:, :, 1:B+1, :] = complex_prod_conj(x0[:, :, 0, None, :].clone(), Tm);
        x0[:, :, B+1:, :] = complex_prod(x0[:, :, 0, None, :].clone(), Tm);
        '''
        assert torch.isnan(x0).any() == False

        out = self.propagate(edge_index=edge_index, x=x0, precomp=precomp, connection=connection)

        return out


    def message(self, x_j, precomp, connection):
        """
        :param x_j: the feature vector of the target neighbours: n_edges x in_channels x B + 1 x 4
        :param precomp: the precomputed part of harmonic networks n_edges x (B+1) x n_rings x 2
        :return: the message from each target to the source nodes n_edges x in_channels x (B + 1) x n_rings x 2 (C)
        """
        
        (E, iC, _, _), R, B = x_j.size(), self.n_rings, self.B

        # Set up result tensors
        res = torch.cuda.FloatTensor(E, iC, 2*B + 1, R, 2).fill_(0)

        # Apply paralell transport (j, i) j --> i
        x_j = complex_prod_conj( x_j.clone(), connection[:, None, None, :] );
        
                     
        # m = 0, +/- 1, ...., B
        
        res[:, :, 0:B+1, :, :] = complex_prod(x_j[:, :, 0:B+1, None, :], precomp[:, None, 0:B+1, :, :])
        
        if (B > 0):
            res[:, :, B+1:, :, :] = complex_prod_conj(x_j[:, :, B+1:, None, :], precomp[:, None, 1:B+1, :, :])
                           
        return res;
        


    def update(self, aggr_out):
        """
        :param aggr_out:  n_nodes x in_channels x 2*B+1 x n_rings x 2
        
        :origin: out_channels x in_channels x n_rings
        :radial: out_channels x in_channels x B x n_rings x 2
        :phase: out_channels x in_channels 
        
        :return: the new feature vector for x: n_nodes x out_channels x 3
        """
        
        (N, iC, _, R, _), oC, B = aggr_out.size(), self.out_channels, self.B
                
        conv =  torch.cuda.FloatTensor(N, oC, iC, B + 1, 2).fill_(0)
                
        conv[:, :, :, 0, :] = torch.sum(torch.mul(aggr_out[:, None, :, 0, :, :], self.real[None, :, :, :, None]), dim=3)
        
        if (B > 0):
            
            conv_plus = complex_prod(aggr_out[:, None, :, 1:(B+1), :, :], self.radial[None, :, :, :, :, :])

            conv_minus = complex_prod_conj(aggr_out[:, None, :, B+1:, :, :], self.radial[None, :, :, :, :, :]);

            conv[:, :, :, 1:, :] =  torch.sum( conv_plus + conv_minus, dim=4);
                
        
        if self.offset:
            
            exp = torch.cat( (torch.cos(self.phase[..., None]), torch.sin(self.phase[..., None])), dim=3)

            conv = complex_prod( conv.clone(), exp[None, ...]);
        
        assert torch.isnan(conv).any() == False
        

        return torch.div(torch.sum(conv, (2, 3)), (2*B + 1) );
        

        
