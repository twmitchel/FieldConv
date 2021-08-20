import torch
import torch.nn as nn
import torch.nn.functional as F

class TwinEval(nn.Module):
    
    def __init__(self, mu=5, ratio=0.5):
        super(TwinEval, self).__init__()
        
        self.mu = mu # Matching threshold value
        self.ratio = ratio;


    def forward(self, xS, xT, p_, n_):
        
        pNorm2 = torch.sum(torch.pow(xT[p_[:, 0], :] - xS[p_[:, 1], :], 2), dim=1)
        nNorm2 = torch.sum(torch.pow(xT[n_[:, 0], :] - xS[n_[:, 1], :], 2), dim=1)
        
        thresh = self.mu * self.ratio
        
        ## Number of false negatives
        nFN = torch.nonzero(pNorm2 > thresh).size(0)
                
        ## Number of false positives
        nFP = torch.nonzero(nNorm2 < thresh).size(0)
                
        return nFN, nFP
        

        
        
    
        
        
        
        

    
