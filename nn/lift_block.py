import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import  TransField, TangentNonLin, TangentPerceptron

class LiftBlock(torch.nn.Module):
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
    def __init__(self, in_channels, out_channels, n_corr_rings=2, offset=True, MLP=True):
        super(LiftBlock, self).__init__()
        
        self.field = TransField(in_channels, out_channels, n_rings = n_corr_rings, offset=offset)
        
        self.nonlin = TangentNonLin(out_channels);
        
        self.MLP = MLP
        
        if self.MLP:
            self.tron1 = TangentPerceptron(out_channels, out_channels)
            self.tron2 = TangentPerceptron(out_channels, out_channels)
            self.tron3 = TangentPerceptron(out_channels, out_channels)



    def forward(self, x, edge_index, pcmp_gather):
                
        # x: n_nodes x in_channels
        # precomp: n_edges x 2 x n_rings x 2
        x = self.nonlin(self.field(x, edge_index, pcmp_gather));
        
        if self.MLP:
            return self.tron3(self.tron2(self.tron1(x)))
        else:
            return x
       # return self.nonlin(self.field(x, edge_index, pcmp_gather));
