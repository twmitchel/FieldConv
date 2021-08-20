import torch
import torch.nn as nn
import torch.nn.functional as F

## Twin loss for source + target features
## Supplement C, Equation (5)

class TwinLoss(nn.Module):
    
    def __init__(self, mu=5):
        super(TwinLoss, self).__init__()
        
        self.mu = mu;


    def forward(self, xS, xT, p_, n_):
        
        numP = p_.size(0) 
        numN = n_.size(0)
        
        out = torch.empty(1, device=xS.device).float().fill_(0)

        lP = torch.sum(torch.pow(xT[p_[:, 0], :] - xS[p_[:, 1], :], 2), (0, 1)) / numP;
        out = out + lP;

        yN = 0.2 * torch.rand( n_.size(0), device=xS.device).float()
        
        lN = torch.sum(torch.pow(xT[n_[:, 0], :] - xS[n_[:, 1], :], 2), dim=1)

        
        lN1 = torch.sum(torch.mul(lN, yN), dim=0);
        
        lN2 = torch.sum(torch.mul(F.relu(torch.mul(torch.ones_like(lN), self.mu) - lN), torch.sub(torch.ones_like(yN), yN)), dim=0)
        out = out + ( (lN1 + lN2) / numN )
            
        
        return out
    
        
        
        
        

    
