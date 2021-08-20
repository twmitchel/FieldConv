import os.path as osp
import shutil
import progressbar
import math

import torch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.io import read_ply, read_off, read_obj
from torch_geometric.nn import fps
import fcutils as fc
import numpy as np
from scipy.io import savemat

import random
import time

class SHREC19(InMemoryDataset):
    
    '''
    The SHREC '19 Isometric and Non-Isometric Shape Correspondence Dataset
    (Dyke et al, 2019). Contains 76 corresponding pairs between 50 meshes
    derived from real-world scans. 
    

    Inputs:
        root (string): Root directory where the dataset should be saved.
        
        which (int): Which set of meshes to load
                    '0': Source meshes in train split
                    '1': Target meshes in train split
                    '2': Source meshes in test split
                    '3': Target meshes in test split
                    
        n_samples (int): Number of sample points on meshes
            
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
    '''

    url = ''
    

    def __init__(self, root, which, n_samples=2048, transform=None, pre_transform=None,
                 pre_filter=None):
        
        self.n_samples = n_samples
        
        super(SHREC19, self).__init__(root, transform, pre_transform, pre_filter)
        
        
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
        
        ## Randomly generate and load test/train split of the models
        fc.splitSHREC19(self.raw_dir)
        pairs = torch.from_numpy(fc.readSplit(self.raw_dir).astype(int)).long();
        
        nSamples = self.n_samples;
        
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
            
            ## We use Pytorch Geometric's Euclidean distance FPS here for convience
            ## In the paper we use a custom FPS which uses the biharmonic distance (to be added in an upcoming release)
            
            sample_idx_S = fps(dataS.pos, batch=None, ratio=(nSamples+1)/ dataS.pos.size(0))
            sample_idx_T = fps(dataT.pos, batch=None, ratio=(nSamples+1)/ dataS.pos.size(0))
            
            sample_idx_S = sample_idx_S[:nSamples].sort()[0]
            sample_idx_T = sample_idx_T[:nSamples].sort()[0]

            
            dataS.sample_idx = sample_idx_S.long();
            dataT.sample_idx = sample_idx_T.long();
                  
            nearest = torch.from_numpy(fc.samplesToNearest(posS, facesS, sample_idx_S.cpu().numpy()).astype(int)).squeeze();
            
            matches = nearest[gt[sample_idx_T]]
               
                                
            ### Source Shape ###

            areaS = fc.surface_area(posS, facesS)

            dataS.pos = torch.div(dataS.pos, np.sqrt(areaS)).float();

            posS = dataS.pos.cpu().numpy();

            ### Target Shape ###

            dataT.pos = torch.div(dataT.pos, np.sqrt(areaS)).float();
            
            posT = dataT.pos.cpu().numpy();            

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
        shutil.rmtree(osp.join(self.raw_dir, 'pairs'))
        shutil.rmtree(osp.join(self.raw_dir, 'gt'))
        
