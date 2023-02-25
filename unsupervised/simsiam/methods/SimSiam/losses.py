import torch.nn.functional as F 

def negcos(p1, p2, z1, z2):
    p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)

    return - 0.5 * ((p1*z2.detach()).sum(dim=1).mean() + (p2*z1.detach()).sum(dim=1).mean())

def k_grad_negcos(p1, p2, z1, z2):
    p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)

    return - 0.5 * ((p1*z2).sum(dim=1).mean() + (p2*z1).sum(dim=1).mean())