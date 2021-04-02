import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import zeros

from nn import ECHO, FieldConv, TangentPerceptron, VectorDropout, TangentLin, TangentNonLin
from utils.field import histDim, norm2D

class ECHOBlock(torch.nn.Module):
    r"""
    ResNet block with convolutions, linearities, and non-linearities
    as described in Harmonic Surface Networks

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
        last_layer (bool, optional): if set to :obj:`True`, does not learn a phase offset
            for the last harmonic conv. (default :obj:`False`)
    """
    def __init__(self, in_channels, n_des, n_bins, out_channels, band_limit=1, n_rings=2, classify=True, mlpC = None):
        super(ECHOBlock, self).__init__()
        
        self.classify = classify
        
        self.conv = FieldConv(in_channels, in_channels, band_limit, n_rings);
        
        self.nonlin = TangentNonLin(in_channels)
        
        self.tron = TangentPerceptron(in_channels, n_des)
           
        self.echo = ECHO(n_des, n_bins, n_des);

        self.hDim = histDim(n_bins)
        
        mid_channels = n_des * self.hDim;
        
        if mlpC is not None:    
            self.lin1 = nn.Linear(mid_channels, mlpC[0])
            self.lin2 = nn.Linear(mlpC[0], mlpC[1]);
            self.lin3 = nn.Linear(mlpC[1], out_channels)
        else:
            self.lin1 = nn.Linear(mid_channels, 256)
            self.lin2 = nn.Linear(256, 128);
            self.lin3 = nn.Linear(128, out_channels)

        self.res = TangentLin(in_channels, out_channels)
                
        
                
    def forward(self, x, edge_index, pcmp_scatter, pcmp_echo, connection):
        
        #print(pcmp_gather.size(), flush=True);
        
        # x: n_nodes x in_channels x 2
        # precomp: n_edges x 2 x n_rings x 2
        
        #print(x.size(), flush=True)
        
        xE = self.tron(self.nonlin(self.conv(x, edge_index, pcmp_scatter, connection)));
                
        xE = self.echo(xE, edge_index, pcmp_echo, connection);
        
        xE = torch.reshape(xE, (xE.size(0), -1))
            
        xE = F.relu(self.lin1(xE))

        xE = F.relu(self.lin2(xE))

        if self.classify:
            return self.lin3(xE) + norm2D(self.res(x))
        else:
            return F.relu(self.lin3(xE) + norm2D(self.res(x)))
