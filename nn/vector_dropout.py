import torch
import torch.nn as nn

class VectorDropout(nn.Module):
    r"""
    Dropout of vector features
    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
    """
    def __init__(self, p=0.5):
        super(VectorDropout, self).__init__()

        self.D = torch.nn.Dropout(p);

    def forward(self, x):
        
        return torch.mul(x, self.D(torch.ones_like(x[..., 0]))[..., None])
