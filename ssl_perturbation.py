import argparse
# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')
# Datasets Options
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
parser.add_argument('--shuffle_train_perturb_data', action='store_true', default=False)
parser.add_argument('--not_shuffle_train_data', action='store_true', default=False)
# Perturbation Options
parser.add_argument('--train_step', default=10, type=int)
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-max', 'random'], help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, help='Perturb type')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='noise shape')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)
# Self-supervised Options
parser.add_argument('--ssl_backbone', default='simclr', type=str, help='Self-supervised backbone')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
parser.add_argument('--min_min_attack_fn', default="non_eot", type=str, help='The function of min_min_attack')
parser.add_argument('--eot_size', default=30, type=int, help='the dimenssion range of test data')
parser.add_argument('--pytorch_aug', action='store_true', default=False)
parser.add_argument('--k_grad', action='store_true', default=False, help='Grandient on both branch')
parser.add_argument('--asymmetric', action='store_true', default=False, help='For moco structure')
parser.add_argument('--moco_t', default=0.1, type=float, help='T for moco')
# CSD Options
parser.add_argument('--ssl_weight', default=1, type=float, help='csd_weight')
parser.add_argument('--linear_noise_csd_weight', default=0, type=float, help='noise_simclr_weight')
parser.add_argument('--linear_noise_csd_index', default=1, type=int, help='noise_simclr_weight')
# Script Options
parser.add_argument('--local_dev', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--job_id', default='', type=str, help='The Slurm JOB ID')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--pre_load_noise_name', default='', type=str, help='Init the random noise')


args = parser.parse_args()

import datetime

import os
if args.local_dev != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local_dev

import shutil
import time
import mlconfig
import toolbox
import torch
# import madrys
import numpy as np
from tqdm import tqdm

import utils
# from utils import train_diff_transform
import datetime
from unsupervised.simclr.simclr_model import Model
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from unsupervised.simclr.simclr_utils import test_ssl, train_simclr
from unsupervised.moco.moco_utils import ModelMoCo, train_moco, test_moco
from unsupervised.simsiam.simsiam_utils import train_simsiam, test_simsiam
from unsupervised.simsiam.methods import set_model
import random

from supervised_models import *

# mlconfig.register(madrys.MadrysLoss)

args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255
flag_shuffle_train_data = not args.not_shuffle_train_data
flag_drop_last = False
if args.ssl_backbone == "moco":
    flag_drop_last = True

# Set up Experiments
if args.load_model_path == '' and args.load_model:
    # args.exp_name = 'exp_' + datetime.datetime.now()
    raise('Use load file name!')

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
utils.build_dirs(exp_path)
utils.build_dirs(checkpoint_path)
logger = utils.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
if not args.no_save:
    logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    if not args.no_save:
        logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
if not args.no_save:
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))

def sample_wise_perturbation(noise_generator, model, optimizer, random_noise, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k):
 
    epochs = args.epochs
    print("The whole epochs are {}".format(epochs))
    if args.job_id == '':
        save_name_pre = 'unlearnable_{}_samplewise_local_{}_{}_{}_{}'.format(args.ssl_backbone, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_{}_samplewise_{}_{}_{}_{}_{}'.format(args.ssl_backbone, args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': []}
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)

    if args.ssl_backbone == 'simclr':
        test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, 0, epochs)
    elif args.ssl_backbone == 'moco':
        test_acc_1, test_acc_5 = test_moco(model.encoder_q, memory_loader, test_loader, 0, epochs, k, args.moco_t, )
    elif args.ssl_backbone == 'simsiam':
        test_acc_1, test_acc_5 = test_simsiam(model.backbone, memory_loader, test_loader, k, temperature, 0, epochs)

    for _epoch_idx in range(1, epochs+1):
        epoch_idx = _epoch_idx
        train_idx = 0
        condition = True
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0, 0

        while condition:
            if args.attack_type == 'min-min':
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    _start = time.time()
                    try:
                        next_item = next(data_iter, end_of_iteration)
                        if next_item != end_of_iteration:
                            (pos_samples_1, pos_samples_2, labels) = next_item
                        else:
                            condition = False
                            del data_iter
                            break
                    except:
                        raise('train loader iteration problem')

                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)

                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        sample_noise = random_noise[label[0].item()]
                        if type(sample_noise) is np.ndarray:
                            mask = sample_noise
                        else:
                            mask = sample_noise.cpu().numpy()
                        sample_noise = torch.from_numpy(mask).to(device)
                        
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)
                        train_idx += 1

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    if args.ssl_backbone == 'simclr':
                        batch_train_loss, batch_size_count, _, _ = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, temperature, pytorch_aug=False)
                    elif args.ssl_backbone == 'moco':
                        batch_train_loss, batch_size_count = train_moco(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer)
                    elif args.ssl_backbone == 'simsiam':
                        batch_train_loss, batch_size_count = train_simsiam(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer)
                    
                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count

                    _end = time.time()

                    # print("traning model time:", _end - _start)

            # Search For Noise
            
            idx = 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)
                # break

                if args.debug:
                    print(torch.max(labels[:, args.linear_noise_csd_index]))

                # Add Sample-wise Noise to each sample
                batch_noise = []
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    sample_noise = random_noise[label[0].item()]
                    if type(sample_noise) is np.ndarray:
                        mask = sample_noise
                    else:
                        mask = sample_noise.cpu().numpy()
                    sample_noise = torch.from_numpy(mask).to(device)
                    batch_noise.append(sample_noise)
                    idx += 1

                # Update sample-wise perturbation
                # model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                batch_noise = torch.stack(batch_noise).to(device)
                if args.attack_type == 'min-min':
                    if args.min_min_attack_fn == "eot_v1":
                        if args.ssl_backbone == 'simclr':
                            eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, random_noise=batch_noise, temperature=temperature, eot_size=args.eot_size, ssl_weight=args.ssl_weight, linear_noise_csd_weight=args.linear_noise_csd_weight, linear_noise_csd_index=args.linear_noise_csd_index)
                        elif args.ssl_backbone == 'moco':
                            eta, train_noise_loss = noise_generator.min_min_attack_moco_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, random_noise=batch_noise, eot_size=args.eot_size, ssl_weight=args.ssl_weight, linear_noise_csd_weight=args.linear_noise_csd_weight, linear_noise_csd_index=args.linear_noise_csd_index, k_grad=args.k_grad)
                        elif args.ssl_backbone == 'simsiam':
                            eta, train_noise_loss = noise_generator.min_min_attack_simsiam_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, random_noise=batch_noise, eot_size=args.eot_size, ssl_weight=args.ssl_weight, linear_noise_csd_weight=args.linear_noise_csd_weight, linear_noise_csd_index=args.linear_noise_csd_index, k_grad=args.k_grad)
                    else:
                        raise('Using wrong min_min_attack_fn in samplewise.')
                else:
                    raise('Invalid attack')

                for delta, label in zip(eta, labels):
                    if torch.is_tensor(random_noise):
                        random_noise[label[0].item()] = delta.detach().cpu().clone()
                    else:
                        random_noise[label[0].item()] = delta.detach().cpu().numpy()

                noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255
        
        train_loss = sum_train_loss / float(sum_train_batch_size)
        results['train_loss'].append(train_loss)
        if args.ssl_backbone == 'simclr':
            test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, 0, epochs)
        elif args.ssl_backbone == 'moco':
            test_acc_1, test_acc_5 = test_moco(model.encoder_q, memory_loader, test_loader, 0, epochs, k, args.moco_t, )
        elif args.ssl_backbone == 'simsiam':
            test_acc_1, test_acc_5 = test_simsiam(model.backbone, memory_loader, test_loader, k, temperature, 0, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['noise_ave_value'].append(noise_ave_value)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['best_loss'].append(best_loss)
        results['best_loss_acc'].append(best_loss_acc)

        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 10 == 0 and not args.no_save:
            if epochs < 100 or epoch_idx % 50 == 0:
                torch.save(model.state_dict(), 'results/{}_checkpoint_model_epoch_{}.pth'.format(save_name_pre, epoch_idx))
                torch.save(random_noise, 'results/{}_checkpoint_perturbation_epoch_{}.pt'.format(save_name_pre, epoch_idx))
                print("model saved at " + save_name_pre)

    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]

            mask = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise, save_name_pre
    else:
        return random_noise, save_name_pre

def main():
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size = args.batch_size

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.train_data_type == 'CIFAR10':
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        train_data.replace_targets_with_id()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=args.num_workers, pin_memory=True, drop_last=flag_drop_last)

        train_noise_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        train_noise_data.replace_targets_with_id()
        train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=args.num_workers, pin_memory=True, drop_last=flag_drop_last)
        # test data don't have to change the target.
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    elif args.train_data_type == 'CIFAR100':
        train_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        train_data.replace_targets_with_id()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=args.num_workers, pin_memory=True, drop_last=flag_drop_last)

        train_noise_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        train_noise_data.replace_targets_with_id()
        train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=args.num_workers, pin_memory=True, drop_last=flag_drop_last)
        # test data don't have to change the target.
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                                num_steps=args.num_steps,
                                                step_size=args.step_size)
    if args.ssl_backbone == "simclr":
        model = Model(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size)
    elif args.ssl_backbone == "moco":
        flag_symmetric = not args.asymmetric
        model = ModelMoCo(dim=128, K=4096, m=0.99, T=args.moco_t, arch='resnet18', bn_splits=8, symmetric=flag_symmetric, )
    elif args.ssl_backbone == "simsiam":
        model = set_model('simsiam', 'resnet18', 'cifar10')
    model = model.cuda()

    if args.load_model:
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        try:
            model.load_state_dict(checkpoints['state_dict'])
        except:
            model.load_state_dict(checkpoints)
        logger.info("File %s loaded!" % (load_model_path))

    if args.ssl_backbone == "simclr":
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    elif args.ssl_backbone == "moco":
        optimizer = optim.SGD(model.parameters(), lr=0.3, weight_decay=1e-4, momentum=0.9)
    elif args.ssl_backbone == "simsiam":
        optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)

    if args.load_model:
        if 'optimizer' in checkpoints:
            optimizer.load_state_dict(checkpoints['optimizer'])
        
    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        if args.random_start:
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape).to(torch.device('cpu'))
            # print(random_noise.device)
        else:
            random_noise = torch.zeros(*args.noise_shape)

        if args.pre_load_noise_name != '':
            random_noise = torch.load("./results/{}.pt".format(args.pre_load_noise_name))
        
        if args.perturb_type == 'samplewise':
            noise, save_name_pre = sample_wise_perturbation(noise_generator, model, optimizer, random_noise, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k)

        else:
            raise('wrong perturb_type')
        torch.save(noise, 'results/{}perturbation.pt'.format(save_name_pre))
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % 'results/{}perturbation.pt'.format(save_name_pre))
    else:
        raise('Not implemented yet')
    return


if __name__ == '__main__':
    # for arg in vars(args):
    #     logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
