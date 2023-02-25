import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18', train_mode='clean_train', f_logits_dim=1024):
        super(Model, self).__init__()

        self.f = []
        self.train_mode = train_mode
        if arch == 'resnet18':
            backbone = resnet18()
            encoder_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            encoder_dim = 2048
        else:
            raise NotImplementedError

        for name, module in backbone.named_children():
            if name == 'conv1' and cifar_head == True:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # encoder
        self.f = nn.Sequential(*self.f)
        # logts for clean_train_softmax
        if self.train_mode == 'clean_train_softmax':
            self.f_logits = nn.Sequential(nn.Linear(encoder_dim, f_logits_dim, bias=True))
        # projection head
        self.g = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        if self.train_mode == 'clean_train_softmax':
            logits = self.f_logits(feature)
            return F.normalize(feature, dim=-1), F.normalize(logits, dim=-1), F.normalize(out, dim=-1)
        else:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        # return feature, F.normalize(out, dim=-1)

class LinearModel(nn.Module):
    def __init__(self, shape, n_class):
        super(LinearModel, self).__init__()
        feature_length = shape[0] * shape[1] * shape[2]
        self.fc = nn.Linear(feature_length, n_class)
    
    def forward(self, x):
        feature = x.reshape(x.shape[0], -1)
        logits = self.fc(feature)
        out = torch.softmax(logits, dim=1)
        
        return logits, out
