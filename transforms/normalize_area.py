import torch
import fcutils as fc
import numpy as np

class NormalizeArea(object):
    '''
    Centers and normalizes to unit surface area.
    
    Adapted from: https://github.com/rubenwiersma/hsn

    '''

    def __init__(self):
        return


    def __call__(self, data):
        
        # Center shapes
        data.pos = data.pos - (torch.max(data.pos, dim=0)[0] + torch.min(data.pos, dim=0)[0]) / 2
        
        # Normalize by surface area
        pos, face = data.pos.cpu().numpy(), data.face.cpu().numpy().T
        area = 1 / np.sqrt(fc.surface_area(pos, face))
        data.pos = data.pos * area

        return data


    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)