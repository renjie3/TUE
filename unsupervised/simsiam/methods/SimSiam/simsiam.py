import torch.nn as nn 

from unsupervised.simsiam.methods.base import CLModel
from .losses import negcos, k_grad_negcos

class SimSiamModel(CLModel):
    def __init__(self, method, arch, dataset):
        super().__init__(method, arch, dataset)
        self.criterion = negcos
        self.k_grad_criterion = k_grad_negcos

        if self.mlp_layers == 2:
            self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),

                    nn.Linear(2048, 2048),
                    nn.BatchNorm1d(2048),
                )

        elif self.mlp_layers == 3:
            self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),

                    nn.Linear(2048, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),

                    nn.Linear(2048, 2048),
                    nn.BatchNorm1d(2048),
                )

        self.pred_head = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 2048),
                )
        
    
    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        # print(x1.shape)
        
        z1 = self.proj_head(x1)
        z2 = self.proj_head(x2)
        
        p1 = self.pred_head(z1)
        p2 = self.pred_head(z2)
        
        return self.criterion(p1, p2, z1, z2)

    def k_grad_forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        # print(x1.shape)
        
        z1 = self.proj_head(x1)
        z2 = self.proj_head(x2)
        
        p1 = self.pred_head(z1)
        p2 = self.pred_head(z2)
        
        return self.k_grad_criterion(p1, p2, z1, z2)
    
