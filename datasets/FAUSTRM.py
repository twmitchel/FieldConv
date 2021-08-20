import os.path as osp
import shutil
import progressbar

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply, read_off

import fcutils as fc
import numpy as np

import random
import time

class FAUSTRM(InMemoryDataset):
    '''
    The remeshed FAUST humans dataset from the paper `Deep Geometric Functional Maps: 
    Robust Feature Learning for Shape Correspondence' (Donati et al. 2020)
    containing 100 watertight meshes representing 10 different poses for 10
    different subjects.
    
    Inputs:
    root (string): Root directory where the dataset should be saved.
        
    train (bool): Whether to load the train or test split
    
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

    
    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        super(FAUSTRM, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[-1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'FAUSTRM.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found.')

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'shapes')
        path = osp.join(path, 'tr_reg_{0:03d}.off')
        label = osp.join(self.raw_dir, 'labels');
        label = osp.join(label, 'tr_reg_{0:03d}.vts');
               
        data_train = [];
        data_test = [];

        # Use 000 vertex labels as template 
        labelZ = np.loadtxt(label.format(0), dtype=np.long);  
        
        for i in progressbar.progressbar(range(100)):
            
            data = read_off(path.format(i))
            pos_tm = data.pos.cpu().numpy();
            faces_tm = data.face.cpu().numpy().T;
            
            area = fc.surface_area(pos_tm, faces_tm)
            
            data.pos = torch.div(data.pos, np.sqrt(area));

            pos_tm = data.pos.cpu().numpy();
            
            labels = np.loadtxt(label.format(i), dtype=np.long); 
            
            labels = fc.composeMap(labelZ, labels, pos_tm, faces_tm)
    
            data.y = torch.squeeze(torch.sub(torch.from_numpy(labels.astype(int)).long(), 1));
        
            data.x = data.pos

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            if i > 79:
                data_test.append(data);
            else:
                data_train.append(data);
              

        torch.save(self.collate(data_train), self.processed_paths[0])
        torch.save(self.collate(data_test), self.processed_paths[1])


        shutil.rmtree(osp.join(self.raw_dir, 'shapes'))
        shutil.rmtree(osp.join(self.raw_dir, 'labels'))
