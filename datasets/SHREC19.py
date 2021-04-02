import os.path as osp
import shutil
import progressbar
import math

import torch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.io import read_ply, read_off, read_obj
from torch_geometric.nn import fps
import trimesh as tm
import vectorheat as vh
import numpy as np
from scipy.io import savemat

import robust_laplacian
import scipy.sparse.linalg as sla

import random
import time

class SHREC19(InMemoryDataset):
    r"""The remeshed FAUST humans dataset from the paper `Multi-directional
    Geodesic Neural Networks via Equivariant Convolution`
    containing 100 watertight meshes representing 10 different poses for 10
    different subjects.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://surfdrive.surf.nl/files/index.php/s/KLSxAN0QEsfJuBV'
    

    def __init__(self, root, which, transform=None, pre_transform=None,
                 pre_filter=None):
        super(SHREC19PR, self).__init__(root, transform, pre_transform, pre_filter)
        
        
        if which == 0:
            path = self.processed_paths[0]
        elif which == 1:
            path = self.processed_paths[1]
        elif which == 2:
            path = self.processed_paths[2]
        else:
            path = self.processed_paths[3]
                
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'SHREC19.zip'

    @property
    def processed_file_names(self):
        return ['train_source.pt', 'train_target.pt', 'test_source.pt', 'test_target.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'models')
        path = osp.join(path, 'scan_{0:03d}.obj')
        path_gt = osp.join(self.raw_dir, 'gt');
        path_gt = osp.join(path_gt, '{0}.{1}.gt.txt');

        scan = 'scan_{0:03d}';
        
        train_source = [];
        train_target = [];
        
        test_source = [];
        test_target = [];
        
        pairs = torch.from_numpy(tm.readSplit().astype(int)).long();
        
        min_rad = 0.02 # above this are null pairs
        nSamples = 2048;
        
        #print(pairs.size(), flush=True)
        #print(pairs, flush=True)
        
        for i in progressbar.progressbar(range(pairs.size(0))):
            
            dataS = read_obj(path.format(pairs[i, 0]))
            dataT = read_obj(path.format(pairs[i, 1]));
            
            dataS.name = scan.format(pairs[i, 0]);
            dataT.name = scan.format(pairs[i, 1]);

            # Read gt file
            gt = torch.from_numpy(np.loadtxt(path_gt.format(scan.format(pairs[i, 0]), scan.format(pairs[i, 1])), dtype=np.long)).long().squeeze();  
            
            ## Sample correspondences 
            posS = dataS.pos.cpu().numpy()
            facesS = dataS.face.cpu().numpy().T
            
            posT = dataT.pos.cpu().numpy();
            facesT = dataT.face.cpu().numpy().T;
            
            sample_idx_S = torch.from_numpy(tm.fpsBiharmonic(posS, facesS, np.arange(dataS.pos.size(0)), nSamples).astype(int)).squeeze();
            
            sample_idx_T = torch.from_numpy(tm.fpsBiharmonic(posT, facesT, np.arange(dataT.pos.size(0)), nSamples).astype(int)).squeeze();
            
            sample_idx_S, _ = torch.sort(sample_idx_S.long());
            sample_idx_T, _ = torch.sort(sample_idx_T.long());
            
            dataS.sample_idx = sample_idx_S.long();
            dataT.sample_idx = sample_idx_T.long();
                  
            nearest = torch.from_numpy(tm.samplesToNearest(posS, facesS, sample_idx_S.cpu().numpy()).astype(int)).squeeze();
            
            matches = nearest[gt[sample_idx_T]]
               
                                
            ### Source Shape ###

            areaS = vh.surface_area(posS, facesS)

            dataS.pos = torch.div(dataS.pos, np.sqrt(areaS)).float();

            posS = dataS.pos.cpu().numpy();

            ### Target Shape ###

            dataT.pos = torch.div(dataT.pos, np.sqrt(areaS)).float();
            
            posT = dataT.pos.cpu().numpy();
            
            #print(dataT.pos.size(), flush=True)
            

            if self.pre_filter is not None and not self.pre_filter(dataT):
                continue
            if self.pre_transform is not None:
                dataT = self.pre_transform(dataT)
          
            
            if self.pre_filter is not None and not self.pre_filter(dataS):
                continue
            if self.pre_transform is not None:
                dataS = self.pre_transform(dataS)
                
            
            pos_pairs = torch.cat( (torch.arange(nSamples)[..., None], matches[..., None]), dim=1);
            
            dataT.pos_pairs = pos_pairs.long()

                    
            
            if pairs[i, 2] == 0:
                train_source.append(dataS);
                train_target.append(dataT);
            else:
                test_source.append(dataS);
                test_target.append(dataT);
            
                
                
      
        torch.save(self.collate(train_source), self.processed_paths[0])
        torch.save(self.collate(train_target), self.processed_paths[1])
        
        torch.save(self.collate(test_source), self.processed_paths[2])
        torch.save(self.collate(test_target), self.processed_paths[3])

        shutil.rmtree(osp.join(self.raw_dir, 'models'))
