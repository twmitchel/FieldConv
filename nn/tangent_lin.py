import torch
import torch.nn as nn

class TangentLin(nn.Module):
    
    '''
    A linear layer for complex tangent vector features. Equivalent to 
    torch.nn.Linear(in_channels, out_channels, bias=False)
    if the module used complex-valued parameters.
    
    No translational offset is applied so as to 
    preserve isometry-equivariance
    '''
    
    def __init__(self, in_channels, out_channels):
        super(TangentLin, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Re = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.Im = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))                            

        torch.nn.init.xavier_uniform_(self.Re)
        torch.nn.init.xavier_uniform_(self.Im, gain=0.1);   
        
    def forward(self, x):
        
        return torch.sum( x[:, None, ...] * torch.complex(self.Re, self.Im)[None, ...], dim=2);
    
