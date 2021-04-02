import torch
import torch.nn.functional as F
import numpy as np
from math import pi as PI
from torch_scatter import scatter_add, scatter_min
from torch_geometric.data import Data


EPS = 1e-12
DELTA = 1e-6;
DELTA_ATAN2 = 1e-6;
PAD = 0.25


## ECHO binning, gets cartesian bins within disk ###
def histDim (n_bins, compact=True):
    
    if not compact:
        return (2*n_bins + 1) * (2*n_bins+1);
    else:
        nHist = 0;
        for i in range(-n_bins, n_bins + 1):
            for j in range(-n_bins, n_bins + 1):
                if ( i * i + j * j <= (n_bins+PAD) * (n_bins+PAD)):
                    nHist = nHist + 1;

        return nHist;

     
def getCentroids(n_bins, compact=True):
    """
    Gets list of histogram bin centroids
    """

    centX = [];
    centY = [];

    for i in range(-n_bins, n_bins + 1):

        for j in range(-n_bins, n_bins + 1):

            if ( i * i + j * j <= (n_bins+PAD) * (n_bins+PAD) and compact):

                centX.append(i);
                centY.append(j);
            else:
                centX.append(i);
                centY.append(j);
            

    return torch.cat( (torch.Tensor(centX)[:, None], torch.FloatTensor(centY)[:, None]), dim=1).long();


def diskInd(n_bins):
    """
    Gets list of histogram bin centroids
    """

    cent = []

    for i in range(0, 2*n_bins + 1):
        for j in range(0, 2*n_bins + 1):
            if  (i - n_bins)*(i - n_bins) + (j - n_bins) * (j - n_bins) <= (n_bins + PAD) * (n_bins + PAD):
                cent.append(i + (2*n_bins + 1)*j)

                
    return torch.cuda.IntTensor(cent);

    
# Evaluation of delta functions
def softBin(x, n_bins):
    # x: n x n_channels x 2
    d = 2 * n_bins + 1
    weights = torch.cuda.FloatTensor(x.size(0), x.size(1), d*d ).fill_(0);
    
    gX, gY = torch.meshgrid(torch.arange(x.size(0), device='cuda'), torch.arange(x.size(1), device='cuda'))
    
    x = torch.mul(x, n_bins);
    
    xC = torch.clamp(torch.ceil(x).long() , -n_bins, n_bins).long()
    xF = torch.clamp(torch.floor(x).long(), -n_bins, n_bins).long()

    #print(torch.prod( xC - x, dim=2 ).size(), flush=True);
    weights[gX, gY,(torch.add(xF[..., 0], n_bins) + d*torch.add(xF[..., 1], n_bins)).long()] = torch.prod( xC - x, dim=2 );
    weights[gX, gY, (torch.add(xC[..., 0], n_bins) + d*torch.add(xC[..., 1], n_bins)).long() ] =  torch.prod(x-xF, dim=2);
    weights[gX, gY, (torch.add(xC[..., 0], n_bins) + d*torch.add(xF[..., 1], n_bins)).long() ] = torch.mul(x[..., 0] - xF[..., 0], xC[..., 1] - x[..., 1]);
    weights[gX, gY, (torch.add(xF[..., 0], n_bins) + d*torch.add(xC[..., 1], n_bins)).long()] = torch.mul(xC[..., 0] - x[..., 0], x[..., 1] - xF[..., 1])


    return weights;


## Tangent vector magnitude and normalization regimes ###

def softNormalize (x, delta=DELTA):
    
    sNorm = softNorm(x, delta);
    
    return torch.div(x, sNorm[..., None]);

def softNorm (x, delta = DELTA):
    
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
    
    return torch.sqrt(torch.add(F.relu(torch.sub(norm2, delta)), delta));


def killNormalize(x, delta=DELTA): # Computers?
    
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
         
    softNorm2 = torch.add(F.relu(torch.sub(norm2, delta)), delta)
    
    killFactor = F.relu(norm2 - delta) / (softNorm2 * torch.sqrt(softNorm2 + delta) );
                 
    return torch.mul(x, killFactor[..., None]);
       
def killNorm(x, delta=DELTA): # Computers?
    
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
         
    softN = softNorm(x, delta)
    
    return F.relu(norm2 - delta) / softN 
                    
def killNorm2(x, delta=DELTA):
                     
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
    
    return F.relu(torch.sub(norm2, delta))      


def softNorm2 (x, delta = DELTA):
    
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
    
    return torch.add(F.relu(torch.sub(norm2, delta)), delta);

def softSignal (x, delta = DELTA):
    
    snorm = softNorm(x, delta);
    
    return torch.sub(snorm, torch.sqrt(torch.mul(torch.ones_like(snorm),  delta)) );

def softAtan2(y, x, delta=DELTA_ATAN2):
    
    return torch.atan2(y, torch.add(F.relu(torch.sub(x, delta)), delta));

    

def norm2D (x):
    
    norm2 = torch.mul(x[..., 0], x[..., 0]) + torch.mul(x[..., 1], x[..., 1]);
    
    return torch.sqrt(torch.max(norm2, EPS*torch.ones_like(norm2)));
        
def normND (x):
    
    norm2 = torch.sum(torch.mul(x, x), dim=-1)
    
    return torch.sqrt(torch.max(norm2, EPS*torch.ones_like(norm2)));

### Complex functions ###

def conj (x):  
    return torch.cat( (x[..., 0, None], -x[..., 1, None]), dim=-1);

def complex_product(a_re, a_im, b_re, b_im):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :param a_re: real component of a
    :param a_im: imaginary component of a
    :param b_re: real component of a
    :param b_im: imaginary component of a
    :return: tuple of real and imaginary components of result
    """
    a_re_ = a_re * b_re - a_im * b_im
    a_im = a_re * b_im + a_im * b_re
    return a_re_, a_im



def complex_prod(a, b):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :return: tuple of real and imaginary components of result
    """
    re = torch.mul(a[..., 0], b[..., 0]) - torch.mul(a[..., 1], b[..., 1])
    im = torch.mul(a[..., 0], b[..., 1]) + torch.mul(a[..., 1], b[..., 0])
    
    return torch.cat( (re[..., None], im[..., None]), dim=-1);


def complex_prod_conj(a, b):
    # a * conj(b)
    
    re = torch.mul(a[..., 0], b[..., 0]) + torch.mul(a[..., 1], b[..., 1]);
    im = torch.mul(a[..., 1], b[..., 0]) - torch.mul(a[..., 0], b[..., 1]);
    
    return torch.cat( (re[..., None], im[..., None]), dim=-1);

def complex_prod_re(a, b):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :param a_re: real component of a
    :param a_im: imaginary component of a
    :param b_re: real component of a
    :param b_im: imaginary component of a
    :return: tuple of real and imaginary components of result
    """
    re = torch.mul(a[..., 0], b[..., 0]) - torch.mul(a[..., 1], b[..., 1])
    return re

def complex_product_re(a_re, a_im, b_re, b_im):
    """
    Computes the complex product of a and b, given the real and imaginary components of both.
    :param a_re: real component of a
    :param a_im: imaginary component of a
    :param b_re: real component of a
    :param b_im: imaginary component of a
    :return: tuple of real and imaginary components of result
    """
    a_re_ = a_re * b_re - a_im * b_im
    return a_re_


def complex_div(a, b):
    
    return torch.div(complex_prod_conj(a, b), softNorm2(b)[..., None]);




### Dataset processing ###
def edge_to_vertex_labels(faces, labels, n_nodes):
    """
    Converts a set of labels for edges to labels for vertices.
    :param faces: face indices of mesh
    :param labels: labels for edges
    :param n_nodes: number of nodes to map to
    """
    edge2key = set()
    edge_index = torch.LongTensor(0, 2)
    for face in faces.transpose(0, 1):
        edges = torch.stack([face[:2], face[1:], face[::2]], dim=0)
        for idx, edge in enumerate(edges):
            edge = edge.sort().values
            edges[idx] = edge
            if tuple(edge.tolist()) not in edge2key:
                edge2key.add(tuple(edge.tolist()))
                edge_index = torch.cat((edge_index, edge.view(1, -1)))

    res = torch.LongTensor(n_nodes).fill_(0)
    res[edge_index[:, 0]] = labels
    res[edge_index[:, 1]] = labels

    return res - 1

def intersection(x, y):      
    combined = torch.cat((torch.unique(x), torch.unique(y)))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts > 1]

def difference(x, y):
    combined = torch.cat((torch.unique(x), torch.unique(y)))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]


## Interpolation weights, we sample on a square root scale

def root_weights(x, n):
        
    weights = torch.zeros((x.size(0), n)).to('cuda');
    
    t = torch.sqrt(torch.div(torch.arange(n), n-1)).to('cuda');
    
    for l in range(n-1):
        
        tInd0 = torch.nonzero(x >= t[l])

        if (l != n-2):
            tInd1 = torch.nonzero(x < t[l+1])
        else:
            tInd1 = torch.nonzero(x <= t[l+1])
            
        tInd = intersection(tInd0, tInd1)
            
        tBar = x[tInd];
        alpha = torch.div(torch.sub(tBar, t[l]), t[l+1] - t[l]);
        
        weights[tInd, l+1] = alpha;
        weights[tInd, l] = torch.ones_like(alpha) - alpha;
    
    return weights;


