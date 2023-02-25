import numpy as np
import torch
from tqdm import tqdm
from utils import train_diff_transform, train_transform_no_totensor
import torch.nn.functional as F

def train_simsiam(net, pos_1, pos_2, train_optimizer):
    net.train()
    total_loss, total_num = 0.0, 0
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)

    loss = net(pos_1, pos_2)
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    total_loss = loss.item()

    return total_loss * pos_1.shape[0], pos_1.shape[0]


def train_simsiam_noise_return_loss_tensor(net, pos_1, pos_2, k_grad=False):
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
    if k_grad:
        loss = net.k_grad_forward(pos_1, pos_2)
    else:
        loss = net(pos_1, pos_2)

    return loss


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_simsiam(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            # print("data.shape:", data.shape)
            # print("feature.shape:", feature.shape)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

        c = np.max(memory_data_loader.dataset.targets) + 1
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
