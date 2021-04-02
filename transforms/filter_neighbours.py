import torch

class FilterNeighbours(object):
    r"""Filters the adjacency matrix and discards neighbours
    that are farther away than the given radius.

    Args:
        radius (float): neighbours have to be within this radius to be maintained.
    """

    def __init__(self, radius):
        self.radius = radius
        return


    def __call__(self, data):
        mask = torch.nonzero(data.edge_attr[:, 0] <= self.radius)[:, 0]

        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]
        if hasattr(data, 'pcmp_scatter'):
            data.pcmp_gather = data.pcmp_gather[mask]
            data.pcmp_scatter = data.pcmp_scatter[mask]
            data.pcmp_echo = data.pcmp_echo[mask]
        if hasattr(data, 'connection'):
            data.connection = data.connection[mask]
        return data
