import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# train num and test num
import random

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_cov.pth.tar')

def train_local(train_loader, model_local, criterion, optimizer, epoch, normalizer):
    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses = AverageMeter()
    mae_errors=AverageMeter()
    
    
    model_local.train()
    
    end = time.time()
    
    for i, ((nbr_fea, atom_bond_num, atom_num), target, batch_cif_ids, atom_table_num) in enumerate(train_loader):
        
        data_time.update(time.time() - end)

        target_normed=normalizer.norm(target)
        target_var=Variable(target_normed)

        atom_bond_fea=Variable(nbr_fea)

        output_local=model_local(atom_bond_fea, atom_num)
        
        ###########################################
#         for parameters in model_local.parameters():
#             print(parameters)
        
#         print(output_local[1])
        #print(output_local[0])
        
        ############################################

        loss=criterion(output_local[0],target_var)
        mae_error = mae(normalizer.denorm(output_local[0].data.cpu()), target)
        
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        #print(i, output_local[0], target_var)
        loss.backward()  ## 反向传播求解梯度

        optimizer.step() ## 更新权重参数
        optimizer.zero_grad() # 清空梯度 非常重要（只要做optimizer.step()，就要接着zero_grad()）

        batch_time.update(time.time() - end)
        end = time.time()

        print_freq=10

        if i%print_freq==0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, mae_errors=mae_errors))


def val_local(val_loader, model_local, criterion, normalizer, test=False):
    
    losses = AverageMeter()
    mae_errors=AverageMeter()
    batch_time=AverageMeter()
    
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        
    model_local.eval()
    
    end = time.time()
    
    for i, ((nbr_fea, atom_bond_num, atom_num), target, batch_cif_ids, atom_table_num) in enumerate(val_loader):

        with torch.no_grad():
            atom_bond_fea=Variable(nbr_fea)
        
        target_normed=normalizer.norm(target)
        with torch.no_grad():
            target_var=Variable(target_normed)
        
        output_local=model_local(atom_bond_fea, atom_num)

        loss=criterion(output_local[0],target_var)
        #print(output_local[0])
        
        mae_error = mae(normalizer.denorm(output_local[0].data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        
        if test:
            test_pred = normalizer.denorm(output_local[0].data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()
        
        print_freq=10


        if i%print_freq==0:
            print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, loss=losses, mae_errors=mae_errors))
        
    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
            
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
        
    return mae_errors.avg

