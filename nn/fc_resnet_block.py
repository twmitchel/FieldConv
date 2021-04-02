import torch
import torch.nn as nn
import torch.nn.functional as F

from nn import FieldConv,  TransField, TangentLin, TangentNonLin

class FCResNetBlock(torch.nn.Module):
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
    
    def __init__(self, in_channels, out_channels, band_limit=1, n_conv_rings=3, offset=True, back=False):
        super(FCResNetBlock, self).__init__()
        

        if not back:
            self.conv1 = FieldConv(in_channels, out_channels, band_limit=band_limit,
                           n_rings=n_conv_rings, offset=offset)
          
            self.conv2 = FieldConv(out_channels, out_channels, band_limit=band_limit,
                           n_rings=n_conv_rings, offset=offset)
        

            self.nonlin1 = TangentNonLin(out_channels);
            self.nonlin2 = TangentNonLin(out_channels);
        else:
            self.conv1 = FieldConv(in_channels, in_channels, band_limit=band_limit,
                           n_rings=n_conv_rings, offset=offset)
                
            self.conv2 = FieldConv(in_channels, out_channels, band_limit=band_limit,
                           n_rings=n_conv_rings, offset=offset)
        

            self.nonlin1 = TangentNonLin(in_channels);
            self.nonlin2 = TangentNonLin(out_channels);
        
        
        self.res = TangentLin(in_channels, out_channels)
        

            
        

    def forward(self, x, edge_index, pcmp_scatter, connection):
        
        #print(pcmp_gather.size(), flush=True);
        
        # x: n_nodes x in_channels x 2
        # precomp_scatter: n_edges x (B+1) x n_rings x 2
        # connection : n_edges x 2
         
        x_conv = self.nonlin1( self.conv1(x, edge_index, pcmp_scatter, connection) );
                                                 
        x_conv = self.conv2(x_conv.clone(), edge_index, pcmp_scatter, connection)
        
        return self.nonlin2(self.res(x) + x_conv);
        
     
    
  