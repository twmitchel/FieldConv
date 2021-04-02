import torch
import torch.nn as nn
from utils.field import complex_prod


class TangentLin(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias=False):
        super(TangentLin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.re = torch.nn.Parameter(torch.Tensor(out_channels, in_channels)).to('cuda')
        self.imag = torch.nn.Parameter(torch.Tensor(out_channels, in_channels)).to('cuda')
        
        self.bias = bias
        
        if self.bias:
            self.trans = torch.nn.Parameter(torch.Tensor(out_channels, 2))
            torch.nn.init.zeros_(self.trans)
        else:
            self.register_parameter('trans', None)

    
        torch.nn.init.xavier_uniform_(self.re)
        torch.nn.init.xavier_uniform_(self.imag, gain=0.1);
                
        self.mat = torch.cat((self.re[..., None], self.imag[..., None]), dim=2).to('cuda')
 
        
        
    def forward(self, x):
        x = torch.sum( complex_prod(self.mat[None, ...], x[:, None, :, :]), dim=2);
        
        if self.bias:
            return x + self.trans[None, ...];
        else:
            return x
    
