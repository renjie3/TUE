from tqdm import tqdm
import torch.nn.functional as F 
import torch
import numpy as np 
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N

@torch.no_grad()
def knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=True, classes=-1, subset=False):
    net.eval()

    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # generate feature bank
    for data, _, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
        feature = net(data.cuda(non_blocking=True))
        feature = F.normalize(feature, dim=1)
        feature_bank.append(feature)
    # feature_bank: [dim, total num]
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    # feature_labels: [total num]
    feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
    
    # loop test data to predict the label by weighted knn search
    test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
    for data, _, target in test_bar:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        feature= net(data)
        feature = F.normalize(feature, dim=1)
        # feature: [bsz, dim]
        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

        total_num += data.size(0)
        total_top1 += (pred_labels[:, 0] == target).float().sum().item()
        test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # feature: [bsz, dim]
    # feature_bank: [dim, total_num]
    # feature_labels: [total_num]
    
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # sim_matrix: [bsz, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    
    # sim_labels: [bsz, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    #one_hot_label: [bsz*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [bsz, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
