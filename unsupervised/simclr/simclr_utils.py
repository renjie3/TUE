import numpy as np
import torch
from tqdm import tqdm
from utils import train_transform_no_totensor, train_diff_transform
import time

def train_simclr(net, pos_1, pos_2, train_optimizer, temperature, pytorch_aug=False):
    net.train()
    total_loss, total_num = 0.0, 0
    if pytorch_aug:
        pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
    else:
        pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    
    out = torch.cat([out_1, out_2], dim=0)
    
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
    pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
    pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
    mask2 = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool()
    
    neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(2 * pos_1.shape[0], -1)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)
    
    sim_weight, sim_indices = neg_sim_matrix2.topk(k=10, dim=-1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    total_loss = loss.item()

    return total_loss * pos_1.shape[0], pos_1.shape[0], torch.log(pos_sim.mean()).item() / 2, torch.log(sim_weight.mean()).item() / 2

# CSD
def get_linear_noise_csd_loss(x, labels):

    sample = x.reshape(x.shape[0], -1)
    cluster_label = labels

    class_center = []
    intra_class_dis = []
    c = torch.max(cluster_label) + 1

    for i in range(c):
        
        idx_i = torch.where(cluster_label == i)[0]
        if idx_i.shape[0] == 0:
            continue
        
        class_i = sample[idx_i, :]
        class_i_center = class_i.mean(dim=0)
        class_center.append(class_i_center)

        point_dis_to_center = torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))
        intra_class_dis.append(torch.mean(point_dis_to_center))

    # print("time3: {}".format(time3 - time2))
    if len(class_center) <= 1:
        return 0
    class_center = torch.stack(class_center, dim=0)

    c = len(intra_class_dis)
    
    class_dis = torch.cdist(class_center, class_center, p=2) # TODO: this can be done for only one time in the whole set

    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

    cluster_DB_loss = ((intra_class_dis_pair_sum + 1e-5) / (class_dis + 1e-5)).mean()
    
    loss = cluster_DB_loss

    # print('get_linear_noise_csd_loss:', cluster_DB_loss.item())

    return loss

def train_simclr_noise_return_loss_tensor(net, pos_1, pos_2, temperature):
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)

    net.eval()

    time0 = time.time()

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    time1 = time.time()

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    time2 = time.time()

    return loss

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_ssl(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
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
            feature, out = net(data)

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
