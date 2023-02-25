import os 
import time 
import torch.nn as nn 
import torch.optim as optim

from warmup_scheduler import GradualWarmupScheduler

# import tensorboard_logger as tb_logger
from unsupervised.simsiam.networks.resnet_org import model_dict 
from unsupervised.simsiam.networks.resnet_cifar import model_dict as model_dict_cifar
from unsupervised.simsiam.simsiam_utils_folder.util import AverageMeter, save_model
from unsupervised.simsiam.simsiam_utils_folder.knn import knn_monitor 

from tqdm import tqdm

class CLModel(nn.Module):
    def __init__(self, method, arch, dataset):
        super().__init__()

        self.method = method
        self.arch = arch 
        self.dataset = dataset

        if 'cifar' in self.dataset:
            print('CIFAR-variant Resnet is loaded')
            model_fun, feat_dim= model_dict_cifar[self.arch]
            self.mlp_layers = 2
        else:
            print('Original Resnet is loaded')
            model_fun, feat_dim = model_dict[self.arch]
            self.mlp_layers = 3
        
        self.model_generator = model_fun
        self.backbone = model_fun()

        self.feat_dim = feat_dim
        
    def forward(self, x):
        pass 
    
    def loss(self, reps):
        pass 

class CLTrainer():
    def __init__(self, args):
        self.args = args 
        # self.tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)
        self.args.warmup_epoch = 10 

    def train(self, model, optimizer, train_loader, test_loader, memory_loader, train_sampler):
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.warmup_epoch, after_scheduler=cosine_scheduler)

        knn_acc = 0.
        for epoch in range(self.args.start_epoch, self.args.epochs):
            model.train()

            losses = AverageMeter()
            cl_losses = AverageMeter()
            
            # if self.args.distributed:
            #     train_sampler.set_epoch(epoch)
            
            optimizer.zero_grad()
            optimizer.step()
            warmup_scheduler.step(epoch)
        
            # 1 epoch training 
            start = time.time()
            for i, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
                v1 = images[0].cuda(self.args.gpu, non_blocking=True)
                v2 = images[1].cuda(self.args.gpu, non_blocking=True)

                # print(len(images))

                # compute representations
                loss = model(v1, v2)
                
                # loss = model.loss(reps)

                losses.update(loss.item(), images[0].size(0))
                cl_losses.update(loss.item(), images[0].size(0))
                    
                # compute gradient and do SGD step
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()

            # KNN-eval 
            if self.args.knn_eval_freq  != 0 and epoch % self.args.knn_eval_freq ==0:
                knn_acc = knn_monitor(model.backbone, memory_loader, test_loader, epoch, classes=self.args.num_classes, subset=self.args.dataset=='imagenet-100')

            print('[{}-epoch] time:{:.3f} | knn acc: {:.3f} | loss:{:.3f} | cl_loss:{:.3f}'.format(epoch+1, time.time() - start, knn_acc, losses.avg, cl_losses.avg))
            
            # Save 
            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename=os.path.join(self.args.saved_path, 'checkpoint.pth.tar'))
            
            print('{}-th epoch saved'.format(epoch+1))
                # save log 
                # self.tb_logger.log_value('train/total_loss', losses.avg, epoch)
                # self.tb_logger.log_value('train/cl_loss', cl_losses.avg, epoch)
                # self.tb_logger.log_value('train/knn_acc', knn_acc, epoch)
                # self.tb_logger.log_value('lr/cnn', optimizer.param_groups[0]['lr'], epoch)
        
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename=os.path.join(self.args.saved_path, 'last.pth.tar'))
        
        print('{}-th epoch saved'.format(epoch+1))