'''Train CIFAR10 with PyTorch.'''
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_class', default=4, type=int, help='num_class')
parser.add_argument('--job_id', default='local', type=str, help='job_id')
parser.add_argument('--load_model', '-r', action='store_true', help='load_model from checkpoint')
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')
parser.add_argument('--just_test', action='store_true', default=False)
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--pre_load_name', default='', type=str, help='load_model_path')
parser.add_argument('--local', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--clean_train', action='store_true', default=False)
parser.add_argument('--samplewise', action='store_true', default=False)
parser.add_argument('--arch', default='resnet18', type=str, help='load_model_path')
parser.add_argument('--train_data_type', default='cifar10', type=str, help='the data used to train')
parser.add_argument('--perturbation_budget', default=1, type=float, help='learning rate')
parser.add_argument('--training_epoch', default=200, type=int, help='num_class')
parser.add_argument('--CA_tmax', default=200, type=int, help='num_class')
args = parser.parse_args()

import os
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from supervised_models import *
# from utils import progress_bar
from tqdm import tqdm
from utils import TransferCIFAR10Pair, CIFAR10Pair, TransferCIFAR100Pair, CIFAR100Pair, plot_loss

import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training
def train(epoch, optimizer):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    train_bar = tqdm(trainloader)
    for pos_1, pos_2, targets in train_bar:
        inputs, targets = pos_1.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.arch in ['resnet18', 'resnet50']:
            feature, outputs = net(inputs)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, args.training_epoch, train_loss/(batch_count), 100.*correct/total))

    return train_loss/(batch_count), 100.*correct/total


def test(epoch, optimizer, save_name_pre):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    test_bar = tqdm(testloader)
    with torch.no_grad():
        for pos_1, pos_2, targets in test_bar:
            inputs, targets = pos_1.to(device), targets.to(device)
            if args.arch in ['resnet18', 'resnet50']:
                feature, outputs = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            batch_count += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, args.training_epoch, test_loss/(batch_count), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # if not args.no_save and not args.just_test:
        #     torch.save(state, './results/{}.pth'.format(save_name_pre))
        best_acc = acc
    return best_acc, test_loss/(batch_count), 100.*correct/total


print ("__name__", __name__)
if __name__ == '__main__':

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    samplewise_perturb = args.samplewise

    if args.train_data_type == 'cifar100':
        args.num_class = 100
    else:
        args.num_class = 10

    if args.pre_load_name == '':
        pre_load_name = None
        save_name_pre = pre_load_name + "_supervised_cleantrain_class{}".format(args.num_class)
    else:
        pre_load_name = args.pre_load_name
        save_name_pre = pre_load_name + "_supervised_class{}_{}".format(args.num_class, args.job_id)

    if args.train_data_type == 'cifar10':
        trainset = TransferCIFAR10Pair(root='data', train=True, transform=transform_train, download=True, perturb_tensor_filepath="./results/{}.pt".format(args.pre_load_name), perturbation_budget=args.perturbation_budget, samplewise_perturb=samplewise_perturb, clean_train=args.clean_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = CIFAR10Pair(root='data', train=False, transform=transform_test, download=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    elif args.train_data_type == 'cifar100':
        trainset = TransferCIFAR100Pair(root='data', train=True, transform=transform_train, download=True, perturb_tensor_filepath="./results/{}.pt".format(args.pre_load_name), perturbation_budget=args.perturbation_budget, samplewise_perturb=samplewise_perturb, clean_train=args.clean_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = CIFAR100Pair(root='data', train=False, transform=transform_test, download=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        

    # Model
    print('==> Building model.. {}'.format(args.arch))

    model_zoo = {'VGG': VGG,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'PreActResNet18': PreActResNet18,
    'GoogLeNet': GoogLeNet,
    'DenseNet121': DenseNet121,
    'ResNeXt29_2x64d': ResNeXt29_2x64d,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DPN92': DPN92,
    'ShuffleNetG2': ShuffleNetG2,
    'SENet18': SENet18,
    # 'ShuffleNetV2': ShuffleNetV2(1),
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF,
    'simpledla': SimpleDLA}

    if 'VGG' in args.arch:
        net = model_zoo['VGG'](args.arch)
    elif args.arch in ['DenseNet121', 'PreActResNet18', 'GoogLeNet']:
        net = model_zoo[args.arch]()
    elif args.arch in ['resnet18'] :
        net = model_zoo[args.arch](args.num_class)
    else:
        net = model_zoo[args.arch]() #(args.num_class)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.load_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./results/{}.pth'.format(args.load_model_path))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.CA_tmax)

    results = {'train_loss': [], 'test_acc': [], 'train_acc': [], 'test_loss': [], 'best_test_acc': []}

    for epoch in range(start_epoch, start_epoch+args.training_epoch):
        train_loss, train_acc = train(epoch, optimizer)
        best_test_acc, test_loss, test_acc = test(epoch, optimizer, save_name_pre)
        scheduler.step()
        # save statistics
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['best_test_acc'].append(best_test_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        # print(epoch, results)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch+2))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        plot_loss('./results/{}_statistics'.format(save_name_pre))
