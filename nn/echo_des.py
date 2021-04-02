from math import pi as PI
import torch
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros

from utils.field import complex_prod_conj, softNormalize, softSignal, softBin, killNorm, diskInd, norm2D

    
class ECHO(MessagePassing):
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
    
    def __init__(self, in_channels, n_bins, out_channels, offset=False, asImage=False, mod=16):
        super(ECHO, self).__init__(aggr='add', flow='source_to_target', node_dim=0)

        self.in_channels = in_channels
        
        self.n_bins = n_bins;
        
        self.disk = diskInd(n_bins).long()
        
        self.asImage = asImage
        
        self.offset = offset;
        
        if self.offset:
            self.phase = torch.nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('phase', None)
            
        self.conn = True if in_channels != out_channels else False
        
        if in_channels != out_channels:
            self.lin = torch.nn.Linear(in_channels, out_channels);
            
        self.mod = mod;
            
        self.reset_parameters
        
        
    def reset_parameters(self):
        glorot(self.phase)  
       

 
    def forward(self, x, edge_index, precomp, connection):
        
        '''
        x: n_nodes x in_channels x 2 {h, T}
        edge_index: 2 x n_edges
        precomp: n_edges x  3 (logarithm, weight)
        
        
        out: n_nodes x out_channels
        '''
        N, iC, m = x.size(0), x.size(1), self.mod
                
        ## Compute decomposition of signal and transformation field
        
        x0 = torch.cuda.FloatTensor(N, iC, 4).fill_(0)

        x0[..., :2] = softNormalize(x);
        x0[..., 2:] = x;
          
        # Sacrifice speed for memory
        if iC <= m:     
            return self.propagate(edge_index=edge_index, x=x0, precomp=precomp, connection = connection)
        else:
            out = self.propagate(edge_index=edge_index, x=x0[:, :m, :], precomp=precomp, connection = connection)
            
            L = torch.floor(torch.Tensor([iC / m])).long()[0]
            r = iC % m;
            
            for l in range(L-1):
                out = torch.cat(  (out, self.propagate(edge_index=edge_index, x=x0[:, m*(l+1):m*(l+2), :], precomp=precomp, connection=connection)), dim=1);
            
            if r > 0:
                out = torch.cat( (out, self.propagate(edge_index=edge_index, x=x0[:, m*L:, :], precomp=precomp, connection=connection)), dim=1);
            
            return out;
    




    def message(self, x_j, precomp, connection):
        """
        :param x_j: the feature vector of the target neighbours: n_edges x in_channels x 3
        :param precomp: the precomputed part of harmonic networks n_edges x 3
        :return: the message from each target to the source nodes n_edges x in_channels x n_bins (C)
        """
        x_j[..., :2] = complex_prod_conj(x_j[..., :2].clone(), precomp[:, None, :2]);
        x_j[..., 2:] = torch.mul(complex_prod_conj( x_j[..., 2:].clone(), connection[:, None, :] ), precomp[:, None, 2, None])
        
        if self.offset:
            exp = torch.cat( (torch.cos(self.phase)[..., None], torch.sin(self.phase)[..., None]), dim=1);
            x_j[..., :2] = complex_prod_conj(x_j[..., :2].clone(), exp[None, ...]);
        
        
        if self.asImage:
            des = torch.reshape(softBin(complex_prod_conj(precomp[:, None, :2], x_j[..., :2]), self.n_bins), (x_j.size(0), x_j.size(1), 2*self.n_bins + 1, 2*self.n_bins + 1) );
        
            return torch.mul(des[..., None], x_j[..., None, None, 2:]);  
        
        else:
            
            des = softBin(complex_prod_conj(precomp[:, None, :2], x_j[..., :2]), self.n_bins)[..., self.disk]
        
            return torch.mul(des[..., None], x_j[..., None, 2:]);  
        
    def update(self, aggr_out):
        """
        :param aggr_out:  n_nodes x in_channels x n_dim x 2

        :filters: out_channels x in_channels x n_param
        
        :return: the new feature vector for x: n_nodes x out_channels x 2
        """ 

        aggr_out = norm2D(aggr_out.clone());

        if self.conn:
            if self.asImage:
                return self.lin(aggr_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2);
            else:
                return self.lin(aggr_out.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return aggr_out

