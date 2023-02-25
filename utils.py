from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
import numpy as np
# from dataset import patch_noise_extend_to_img

import random
import matplotlib.pyplot as plt

import kornia.augmentation as Kaug
import torch.nn as nn
import os
import pickle
from typing import Any, Callable, Optional, Tuple
import pandas as pd

import math
import logging

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

ToTensor_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_diff_transform = nn.Sequential(
    Kaug.RandomResizedCrop([32,32]),
    Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    Kaug.RandomGrayscale(p=0.2)
)

train_transform_no_totensor = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # transforms.ToTensor(),
    ])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display

def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_pairs_of_imgs(idx, clean_train_dataset, noise, samplewise = False):
    clean_img = clean_train_dataset.data[idx]
    clean_img = transforms.functional.to_tensor(clean_img)
    if samplewise:
        unlearnable_img = torch.clamp(clean_img + noise[idx].cpu(), 0, 1)

        x = noise[idx].cpu()
    else:
        unlearnable_img = torch.clamp(clean_img + noise[clean_train_dataset.targets[idx]], 0, 1)

        x = noise[clean_train_dataset.targets[idx]]
    x_min = torch.min(x)
    x_max = torch.max(x)
    noise_norm = (x - x_min) / (x_max - x_min)
    noise_norm = torch.clamp(noise_norm, 0, 1)

    return [clean_img, noise_norm, unlearnable_img]

def save_img_group(clean_train_dataset, noise, img_path, samplewise = False):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    selected_idx = [random.randint(0, 127) for _ in range(9)]
    img_grid = []
    for idx in selected_idx:
        img_grid += get_pairs_of_imgs(idx, clean_train_dataset, noise, samplewise)

    img_grid_tensor = torchvision.utils.make_grid(torch.stack(img_grid), nrow=9, pad_value=255)
    npimg = img_grid_tensor.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)

def save_img_group_by_index(clean_train_dataset, noise, img_path, selected_idx, samplewise = False):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    if len(selected_idx) != 9:
        raise("Please use 9 indexes")
    # selected_idx = [random.randint(0, 127) for _ in range(9)]
    img_grid = []
    for idx in selected_idx:
        img_grid += get_pairs_of_imgs(idx, clean_train_dataset, noise, samplewise)

    img_grid_tensor = torchvision.utils.make_grid(torch.stack(img_grid), nrow=9, pad_value=255)
    npimg = img_grid_tensor.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)

def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask


class TransferSVHNPair(SVHN):
    """SVHN Dataset.
    """
    def __init__(self, root='data', train='train', transform=None, download=True, perturb_tensor_filepath=None, perturbation_budget=1.0, samplewise_perturb: bool = False, flag_save_img_group: bool = False, perturb_rate: float = 1.0, clean_train=False, small: bool = False, noise_small_file = False):
        super(TransferSVHNPair, self).__init__(root=root, split=train, download=download, transform=transform)

        self.samplewise_perturb = samplewise_perturb

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None
            return
        
        self.perturbation_budget = perturbation_budget

        self.data = np.transpose(self.data, [0, 2, 3, 1])
        # print(self.labels.shape)
        # input('done')

        label_counter = [0 for _ in range(10)]
        new_data = []
        new_labels = []
        new_noise = []

        if train == 'train':
            if small:
                for i, label in enumerate(self.labels):
                    if label_counter[label] < 5000:
                        new_data.append(self.data[i])
                        if not noise_small_file:
                            new_noise.append(self.noise_255[i])
                        new_labels.append(label)
                        label_counter[label] += 1

                self.data = np.stack(new_data, axis=0)
                self.labels = np.array(new_labels)

                # with open('./small_svhn_train_targets.pkl', "wb") as f:
                #     pickle.dump(self.labels, f)
                # input('save_svhn done')
                
                if not noise_small_file:
                    self.noise_255 = np.stack(new_noise, axis=0)

        if not clean_train:
            if not flag_save_img_group:
                perturb_rate_index = np.random.choice(len(self.labels), int(len(self.labels) * perturb_rate), replace=False)
                self.data = self.data.astype(np.float32)
                for idx in range(len(self.data)):
                    if idx not in perturb_rate_index:
                        continue
                    if not samplewise_perturb:
                        # raise('class_wise still under development')
                        noise = self.noise_255[self.labels[idx]]
                    else:
                        noise = self.noise_255[idx]
                        # print("check it goes samplewise.")
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
                    # noise = np.transpose(noise, [2, 0, 1])
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print('load perturb done______________________________')
        else:
            print('it is clean train')

        # print(self.data.shape)
        # input("check")


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # print(img[0][0])
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            pos_1 = torch.clamp(self.transform(img), 0, 1)
            pos_2 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def save_noise_img(self):
        if self.class_4:
            class_num = 4
        else:
            class_num = 10

        np_targets = np.array(self.targets)
        mean_one_class = []
        for i in range(class_num):
            one_class_index = np.where(np_targets == i)[0]
            noise_one_class = self.noise_255[one_class_index]
            mean_one_class.append(noise_one_class.mean(axis=0))
            for j in range(len(one_class_index) // 9):
                save_img_group_by_index(self, self.perturb_tensor, "./test.png", one_class_index[j*9:(j+1)*9], self.samplewise_perturb)
                cmd = input()
                if cmd == 'next':
                    break

class SVHNPair(SVHN):
    """SVHN Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        small: bool = False,
    ) -> None:

        super(SVHNPair, self).__init__(root, split=train, transform=transform, target_transform=target_transform, download=download)
        # self.train_noise_after_transform = train_noise_after_transform

        self.data = np.transpose(self.data, [0, 2, 3, 1])
        print(self.labels.shape)

        # with open('./large_svhn_train_targets.pkl', "wb") as f:
        #     pickle.dump(self.labels, f)
        # input('save_svhn done')

        label_counter = [0 for _ in range(10)]
        new_data = []
        new_labels = []

        if train == 'train':
            if small:
                for i, label in enumerate(self.labels):
                    if label_counter[label] < 5000:
                        new_data.append(self.data[i])
                        new_labels.append(label)
                        label_counter[label] += 1

                self.data = np.stack(new_data, axis=0)
                self.labels = np.array(new_labels)


    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_targets_with_id(self):
        idx_label = []
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            idx_label.append(i)

        gt_label = np.array(self.targets)
        idx_label = np.array(idx_label)
        # print(gt_label.shape)
        # print(idx_label.shape)
        if len(gt_label.shape) > 1:
            idx_label = np.expand_dims(idx_label, axis=1)
            self.targets = np.concatenate([idx_label, gt_label], axis=1)
        else:
            self.targets = np.stack([idx_label, gt_label], axis=1)

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CIFAR10Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_targets_with_id(self):
        idx_label = []
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            idx_label.append(i)

        gt_label = np.array(self.targets)
        idx_label = np.array(idx_label)

        if len(gt_label.shape) > 1:
            idx_label = np.expand_dims(idx_label, axis=1)
            self.targets = np.concatenate([idx_label, gt_label], axis=1)
        else:
            self.targets = np.stack([idx_label, gt_label], axis=1)

class CIFAR10_linear(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        use_sub_test = False,
        max_sub_test_label_idx = False,
    ) -> None:

        super(CIFAR10_linear, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if use_sub_test and train == False:
            self.targets = np.array(self.targets)
            target_mask_idx = np.zeros(len(self.targets))

            for i in range(max_sub_test_label_idx):
                # print(i, np.sum(target_mask_idx))
                target_mask_idx[self.targets == i] = 1
            self.targets = self.targets[target_mask_idx == 1]
            self.data = self.data[target_mask_idx == 1]

class CIFAR100_linear(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        use_sub_test = False,
        max_sub_test_label_idx = False,
    ) -> None:

        super(CIFAR100_linear, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if use_sub_test and train == False:
            self.targets = np.array(self.targets)
            target_mask_idx = np.zeros(len(self.targets))

            for i in range(max_sub_test_label_idx):
                # print(i, np.sum(target_mask_idx))
                target_mask_idx[self.targets == i] = 1
            self.targets = self.targets[target_mask_idx == 1]
            self.data = self.data[target_mask_idx == 1]

class CIFAR10PairTuple(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        class_4: bool = True,
        train_noise_after_transform: bool = True,
        mix: str = 'no', 
        gray: str = 'no', 
        class_4_train_size = 1024,
        kmeans_index = -1,
        kmeans_index2 = -1,
        unlearnable_kmeans_label = False,
        kmeans_label_file = ''
    ) -> None:

        super(CIFAR10PairTuple, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train
        
        self.train_noise_after_transform = train_noise_after_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train:
                if self.train_noise_after_transform:
                    pos_1 = train_transform(img)
                    pos_2 = train_transform(img)
                else:
                    pos_1 = self.transform(img)
                    pos_2 = self.transform(img)
            else:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [pos_1, pos_2], target

    def replace_targets_with_id(self):
        idx_label = []
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            idx_label.append(i)

        gt_label = np.array(self.targets)
        idx_label = np.array(idx_label)
        # print(gt_label.shape)
        # print(idx_label.shape)
        if len(gt_label.shape) > 1:
            idx_label = np.expand_dims(idx_label, axis=1)
            self.targets = np.concatenate([idx_label, gt_label], axis=1)
        else:
            self.targets = np.stack([idx_label, gt_label], axis=1)


class CIFAR100PairTuple(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        kmeans_index = -1,
        kmeans_label_file = ''
    ) -> None:

        super(CIFAR100PairTuple, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train

        if kmeans_index >= 0:
            if kmeans_label_file == '':
                raise('use kmeans_label_file')
            else:
                kmeans_filepath = os.path.join(root, "kmeans_label/{}.pkl".format(kmeans_label_file))
            with open(kmeans_filepath, "rb") as f:
                kmeans_labels = pickle.load(f)[kmeans_index]
                print("kmeans_label_num: ", np.max(kmeans_labels)+1)

            self.targets = kmeans_labels
        
        # self.train_noise_after_transform = train_noise_after_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)
            else:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [pos_1, pos_2], target

    def replace_targets_with_id(self):
        idx_label = []
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            idx_label.append(i)

        gt_label = np.array(self.targets)
        idx_label = np.array(idx_label)
        # print(gt_label.shape)
        # print(idx_label.shape)
        if len(gt_label.shape) > 1:
            idx_label = np.expand_dims(idx_label, axis=1)
            self.targets = np.concatenate([idx_label, gt_label], axis=1)
        else:
            self.targets = np.stack([idx_label, gt_label], axis=1)


class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CIFAR100Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_targets_with_id(self):
        idx_label = []
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            idx_label.append(i)

        gt_label = np.array(self.targets)
        idx_label = np.array(idx_label)
        # print(gt_label.shape)
        # print(idx_label.shape)
        self.targets = np.stack([idx_label, gt_label], axis=1)

class TransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0, class_4: bool = True, samplewise_perturb: bool = False, org_label_flag: bool = False, flag_save_img_group: bool = False, perturb_rate: float = 1.0, clean_train=False, kmeans_index=-1, unlearnable_kmeans_label=False, kmeans_label_file='', in_tuple=False, flag_perturbation_budget=False):
        super(TransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        self.class_4 = class_4
        self.samplewise_perturb = samplewise_perturb
        self.in_tuple = in_tuple

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            # print(self.perturb_tensor)
            # print(self.perturb_tensor.shape)
            # input()
            if flag_perturbation_budget:
                self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-255*perturbation_budget, 255*perturbation_budget).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-9, 9).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None
            return
        
        self.perturbation_budget = perturbation_budget

        if not clean_train:
            if not flag_save_img_group:
                perturb_rate_index = np.random.choice(len(self.targets), int(len(self.targets) * perturb_rate), replace=False)
                self.data = self.data.astype(np.float32)
                for idx in range(len(self.data)):
                    if idx not in perturb_rate_index:
                        continue
                    if not samplewise_perturb:
                        # raise('class_wise still under development')
                        noise = self.noise_255[self.targets[idx]]
                        # if org_label_flag:
                        #     noise = self.noise_255[self.targets[idx]]
                        # else:
                        #     noise = self.noise_255[self.random_noise_class[idx]]
                    else:
                        noise = self.noise_255[idx]
                        # print("check it goes samplewise.")
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print('load perturb done______________________________')
        else:
            print('it is clean train')


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = torch.clamp(self.transform(img), 0, 1)
            pos_2 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.in_tuple:
            return [pos_1, pos_2], target
        else:
            return pos_1, pos_2, target

    def save_noise_img(self):
        if self.class_4:
            class_num = 4
        else:
            class_num = 10

        np_targets = np.array(self.targets)
        mean_one_class = []
        for i in range(class_num):
            one_class_index = np.where(np_targets == i)[0]
            noise_one_class = self.noise_255[one_class_index]
            mean_one_class.append(noise_one_class.mean(axis=0))
            for j in range(len(one_class_index) // 9):
                save_img_group_by_index(self, self.perturb_tensor, "./test.png", one_class_index[j*9:(j+1)*9], self.samplewise_perturb)
                cmd = input()
                if cmd == 'next':
                    break

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0, class_4: bool = True, samplewise_perturb: bool = False, org_label_flag: bool = False, flag_save_img_group: bool = False, perturb_rate: float = 1.0, clean_train=False, kmeans_index=-1, unlearnable_kmeans_label=False):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        self.class_4 = class_4
        self.samplewise_perturb = samplewise_perturb

        if class_4:
            sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
            with open(sampled_filepath, "rb") as f:
                sampled_data = pickle.load(f)
            if train:
                self.data = sampled_data["train_data"]
                self.targets = sampled_data["train_targets"]
            else:
                self.data = sampled_data["test_data"]
                self.targets = sampled_data["test_targets"]

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None
            return

        if random_noise_class_path != None:
            self.random_noise_class = np.load(random_noise_class_path)
        else:
            self.random_noise_class = None
        
        self.perturbation_budget = perturbation_budget

    # random_noise_class = np.load('noise_class_label.npy')
        # self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()

        if not clean_train:
            perturb_num = [0 for _ in range(10)]
            if not flag_save_img_group:
                perturb_rate_index = np.arange(int(len(self.targets) * perturb_rate))
                self.data = self.data.astype(np.float32)
                for idx in range(len(self.data)):
                    if idx not in perturb_rate_index:
                        continue
                    if not samplewise_perturb:
                        # raise('class_wise still under development')
                        noise = self.noise_255[self.targets[idx]]
                        # if org_label_flag:
                        #     noise = self.noise_255[self.targets[idx]]
                        # else:
                        #     noise = self.noise_255[self.random_noise_class[idx]]
                    else:
                        noise = self.noise_255[idx]
                        perturb_num[self.targets[idx]] += 1
                        # print("check it goes samplewise.")
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            # print(perturb_num)
            # input()
        else:
            print('it is clean train')


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = torch.clamp(self.transform(img), 0, 1)
            pos_2 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def save_noise_img(self):
        if self.class_4:
            class_num = 4
        else:
            class_num = 10

        np_targets = np.array(self.targets)
        mean_one_class = []
        for i in range(class_num):
            one_class_index = np.where(np_targets == i)[0]
            noise_one_class = self.noise_255[one_class_index]
            mean_one_class.append(noise_one_class.mean(axis=0))
            for j in range(len(one_class_index) // 9):
                save_img_group_by_index(self, self.perturb_tensor, "./test.png", one_class_index[j*9:(j+1)*9], self.samplewise_perturb)
                cmd = input()
                if cmd == 'next':
                    break

class TransferCIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0, samplewise_perturb: bool = False, org_label_flag: bool = False, flag_save_img_group: bool = False, perturb_rate: float = 1.0, clean_train=False, kmeans_index=-1, unlearnable_kmeans_label=False, kmeans_label_file='', in_tuple=False, flag_perturbation_budget=False):
        super(TransferCIFAR100Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        self.samplewise_perturb = samplewise_perturb
        self.in_tuple = in_tuple

        # if train:
        #     with open('./cifar100_train_targets.pkl', "wb") as f:
        #         pickle.dump(self.targets, f)
        # else:
        #     with open('./cifar100_test_targets.pkl', "wb") as f:
        #         pickle.dump(self.targets, f)
        
        # input('save_targets done')

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            if flag_perturbation_budget:
                self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-255*perturbation_budget, 255*perturbation_budget).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-9, 9).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None
            if kmeans_index >= 0:
                if kmeans_label_file == '':
                    if class_4:
                        kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_4class.pkl")
                    else:
                        if not unlearnable_kmeans_label:
                            kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_cifar10.pkl")
                        else:
                            kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_unlearnable_simclr_label.pkl")
                else:
                    kmeans_filepath = os.path.join(root, "kmeans_label/{}.pkl".format(kmeans_label_file))
                with open(kmeans_filepath, "rb") as f:
                    kmeans_labels = pickle.load(f)[kmeans_index]
                    print(kmeans_filepath)
                    # print(kmeans_labels[:100])
                    # input()
                    print("kmeans_label_num: ", np.max(kmeans_labels)+1)

                self.targets = kmeans_labels
            return

        if random_noise_class_path != None:
            self.random_noise_class = np.load(random_noise_class_path)
        else:
            self.random_noise_class = None
        
        self.perturbation_budget = perturbation_budget

        if not clean_train:
            if not flag_save_img_group:
                perturb_rate_index = np.random.choice(len(self.targets), int(len(self.targets) * perturb_rate), replace=False)
                self.data = self.data.astype(np.float32)
                for idx in range(len(self.data)):
                    if idx not in perturb_rate_index:
                        continue
                    if not samplewise_perturb:
                        # raise('class_wise still under development')
                        noise = self.noise_255[self.targets[idx]]
                        # if org_label_flag:
                        #     noise = self.noise_255[self.targets[idx]]
                        # else:
                        #     noise = self.noise_255[self.random_noise_class[idx]]
                    else:
                        noise = self.noise_255[idx]
                        # print("check it goes samplewise.")
                    # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
            print('load perturb done ______________')
        else:
            print('it is clean train')

        if kmeans_index >= 0:
            if kmeans_label_file == '':
                if class_4:
                    kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_4class.pkl")
                else:
                    if not unlearnable_kmeans_label:
                        kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_cifar10.pkl")
                    else:
                        kmeans_filepath = os.path.join(root, "kmeans_label/kmeans_unlearnable_simclr_label.pkl")
            else:
                kmeans_filepath = os.path.join(root, "kmeans_label/{}.pkl".format(kmeans_label_file))
            with open(kmeans_filepath, "rb") as f:
                kmeans_labels = pickle.load(f)[kmeans_index]
                print(kmeans_filepath)
                print("kmeans_label_num: ", np.max(kmeans_labels)+1)

            self.targets = kmeans_labels


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img[0][0])
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            # print(self.perturb_tensor[self.random_noise_class[index]][0][0])
            # print("self.transform(img)", self.transform(img).shape)
            # pos_1 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)
            # pos_2 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)
            pos_1 = torch.clamp(self.transform(img), 0, 1)
            pos_2 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.in_tuple:
            return [pos_1, pos_2], target
        else:
            return pos_1, pos_2, target

    def save_noise_img(self):
        if self.class_4:
            class_num = 4
        else:
            class_num = 10

        np_targets = np.array(self.targets)
        mean_one_class = []
        for i in range(class_num):
            one_class_index = np.where(np_targets == i)[0]
            noise_one_class = self.noise_255[one_class_index]
            mean_one_class.append(noise_one_class.mean(axis=0))
            for j in range(len(one_class_index) // 9):
                save_img_group_by_index(self, self.perturb_tensor, "./test.png", one_class_index[j*9:(j+1)*9], self.samplewise_perturb)
                cmd = input()
                if cmd == 'next':
                    break



def plot_loss(file_prename):
    pd_reader = pd.read_csv(file_prename+".csv")
    # print(pd_reader)

    epoch = pd_reader.values[:,0]
    loss = pd_reader.values[:,1]
    acc = pd_reader.values[:,2]

    fig, ax=plt.subplots(1,1,figsize=(9,6))
    ax1 = ax.twinx()

    p2 = ax.plot(epoch, loss,'r-', label = 'loss')
    ax.legend()
    p3 = ax1.plot(epoch,acc, 'b-', label = 'test_acc')
    ax1.legend()

    #显示图例
    # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
    # plt.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax1.set_ylabel('acc')
    plt.title('Training loss on generating model and clean test acc')
    plt.savefig(file_prename + ".png")
    


def train_supervised_batch(g_net, pos_1, targets, supervised_criterion, supervised_optimizer, supervised_transform_train):
    
    supervised_optimizer.zero_grad()
    inputs = supervised_transform_train(pos_1)
    feature, outputs = g_net(inputs)
    loss = supervised_criterion(outputs, targets)
    loss.backward()
    supervised_optimizer.step()

    return loss.item()

def plot_tsne(feature_bank, GT_label, save_name_pre):

    labels = GT_label
    if torch.is_tensor(labels):
        plot_labels_colar = labels.detach().cpu().numpy()
    else:
        plot_labels_colar = labels
    c = np.max(plot_labels_colar) + 1

    print(c)
    feature_tsne_output = feature_bank
        
    coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
    coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

    cm = plt.cm.get_cmap('gist_rainbow', c)

    marker = ['o', 'x', 'v', 'd']
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'chartreuse', 'cyan', 'sage', 'coral', 'gold', 'plum', 'sienna', 'teal']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.title("\n max:{} min:{}".format(coord_max, coord_min))

    x_pos_1 = feature_tsne_output[:, 0]
    y_pos_1 = feature_tsne_output[:, 1]
    # plot_labels_colar = labels.detach().cpu().numpy()

    # linewidths

    aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar, cmap=cm)

    plt.xlim((coord_min, coord_max))
    plt.ylim((coord_min, coord_max))
    if not os.path.exists('./plot_feature/{}'.format(save_name_pre)):
        os.mkdir('./plot_feature/{}'.format(save_name_pre))
    plt.savefig('./plot_feature/{}/{}_{}.png'.format(save_name_pre, c, save_name_pre))
    plt.close()


def get_centers(feature, labels, use_normalized):
    sample = feature.reshape(feature.shape[0], -1)
    c = np.max(labels) + 1
    centroids = []
    for i in range(c):
        idx_i = np.where(labels == i)[0]
        if idx_i.shape[0] == 0:
            raise('wrong here')
            continue
        class_i = sample[idx_i, :]
        if use_normalized:
            class_i_center = nn.functional.normalize(class_i.mean(dim=0), p=2, dim=0)
        else:
            class_i_center = class_i.mean(dim=0)
        centroids.append(class_i_center)

    return centroids
        