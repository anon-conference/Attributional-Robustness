# -*- coding: utf-8 -*-
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils

from network import resnet
from tensorboardX import SummaryWriter
from utils_adl.util_args import get_args
from utils_adl.util_acc import accuracy, adjust_learning_rate, \
    save_checkpoint, AverageEpochMeter, SumEpochMeter, \
    ProgressEpochMeter, calculate_IOU, Logger
from utils_adl.util_loader import data_loader
from utils_adl.util_bbox import *
from utils_adl.util_cam import *
from utils_adl.util_eval import *
from utils_adl.util import *
from saliency_pgd import * 

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


best_epoch = 0
best_acc1 = 0
best_loc1 = 0
loc1_at_best_acc1 = 0
acc1_at_best_loc1 = 0
gtknown_at_best_acc1 = 0
gtknown_at_best_loc1 = 0
writer = None

print("process id ", os.getpid())

ranking_loss = nn.SoftMarginLoss()
cos = nn.CosineSimilarity(dim=1, eps=1e-15)



def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    global best_acc1, best_loc1, best_epoch, \
        loc1_at_best_acc1, acc1_at_best_loc1, \
        gtknown_at_best_acc1, gtknown_at_best_loc1
    global writer

    args.gpu = 0
    log_folder = os.path.join('train_log', args.name)

    if args.gpu == 0:
        writer = SummaryWriter(logdir=log_folder)

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    Logger(os.path.join(log_folder, 'log.log'))
    print(args.gpu)
    
    if args.dataset == 'CUB':
        num_classes = 200
    elif args.dataset == 'ILSVRC':
        num_classes = 1000
    else:
        raise Exception("Not preferred dataset.")

    model = resnet.resnet50(pretrained=True,beta_value = 100. ,
                                num_classes=num_classes)
   
    model = torch.nn.DataParallel(model,device_ids = range(torch.cuda.device_count()) ).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    param_features = []
    param_classifiers = []
    
    for name, parameter in model.named_parameters():
        if 'layer4.' in name or 'fc.' in name:
            param_classifiers.append(parameter)
        else:
            param_features.append(parameter)
    
    optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': args.lr},
            {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}],
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nest)

    # optionally resume from a checkpoint
    if args.resume:
        print(args.resume)
        model, optimizer = load_model(model, optimizer, args)
        
        
    cudnn.benchmark = True

    # CUB-200-2011
    train_loader, val_loader, train_sampler = data_loader(args)

    if args.cam_curve:
        cam_curve(val_loader, model, criterion, writer, args)
        return

    if args.evaluate:
        evaluate(val_loader, model, criterion, args)
        return

    if args.gpu == 0:
        print("Batch Size per Tower: %d"%(args.batch_size))
        print(model)
    
    print("saliency training started")
    config = {
            'epsilon': 2.0 / 255.0,
            'num_steps': 3,
            'step_size': 1.5/ 255.0,
            'k_top': 15000,
            'img_size':224,
            'n_classes': num_classes
        
            }
    
    print(config)
    mean = torch.tensor([.485, .456, .406])
    std = torch.tensor([.229, .224, .225])
    
    def normalize(x):
        new_std = std[..., None, None]
        new_mean = mean[..., None, None]
        return (x - new_mean.cuda())/new_std.cuda()
    
    attack = AttackSaliency(config , normalize)
    
    conv2d = nn.Conv2d(1, 1, kernel_size=(3,3), bias=False , stride = (1,1),padding = 1)
    conv2d.weight = torch.nn.Parameter( (1./9.)*torch.ones((1,1,3,3)).cuda())
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.gpu == 0:
            print("===========================================================")
            print("Start Epoch %d ..." % (epoch+1))

        
        adjust_learning_rate(optimizer, epoch, args)
        val_acc1 = 0
        val_loss = 0
        val_gtloc_g = 0
        val_loc_g = 0
     
        # train for one epoch
        torch.cuda.empty_cache()
        
        train_acc, train_loss, progress_train = \
                train(train_loader, model, criterion, optimizer, epoch, args , attack , num_classes)

        if args.gpu == 0:
            progress_train.display(epoch+1)

        # evaluate on validation set
        if args.task == 'cls':
            val_acc1, val_loss = validate(val_loader, model, criterion, epoch, args)

        # evaluate localization on validation set
        elif args.task == 'wsol':
#             val_acc1, val_acc5, val_loss, \
#             val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, epoch, args)
            val_acc1, val_acc5, val_loss, \
            val_gtloc_g, val_loc_g = evaluate_loc_grad(val_loader, model, normalize, criterion, epoch, args,conv2d)

        # tensorboard
        if args.gpu == 0:
            writer.add_scalar(args.name + '/train_acc', train_acc, epoch)
            writer.add_scalar(args.name + '/train_loss', train_loss, epoch)
            writer.add_scalar(args.name + '/val_cls_acc', val_acc1, epoch)
            writer.add_scalar(args.name + '/val_loss', val_loss, epoch)
            writer.add_scalar(args.name + '/val_gt_loc', val_gtloc_g, epoch)
            writer.add_scalar(args.name + '/val_loc1', val_loc_g, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if is_best:
            best_epoch = epoch + 1
            loc1_at_best_acc1 = val_loc_g
            gtknown_at_best_acc1 = val_gtloc_g

        if args.task == 'wsol':
            # in case best loc,, Not using this.
            is_best_loc = val_loc_g > best_loc1
            best_loc1 = max(val_loc_g, best_loc1)
            if is_best_loc:
                acc1_at_best_loc1 = val_acc1
                gtknown_at_best_loc1 = val_gtloc_g

        if args.gpu == 0:
            print("\nCurrent Best Epoch: %d" %(best_epoch))
            print("Top-1 GT-Known Localization Acc: %.3f \
                   \nTop-1 Localization Acc: %.3f\
                   \nTop-1 Classification Acc: %.3f" % \
                  (gtknown_at_best_acc1, loc1_at_best_acc1, best_acc1))
            print("\nEpoch %d finished." % (epoch+1))

        saving_dir = os.path.join(log_folder)
        
        save_dir = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }
            
        save_checkpoint(save_dir, is_best, saving_dir)
        
        save_checkpoint(save_dir, is_best_loc, saving_dir,filename= 'best_loc_checkpoint.pth.tar')
        
        
        
        save_freq = 10
        if ((epoch % save_freq==0) or (epoch==args.epochs-1)) :
            save_checkpoint(save_dir, False, saving_dir,filename= 'checkpoint_' + str(epoch) + '.pth.tar')


    if args.gpu == 0:
        save_train(best_acc1, loc1_at_best_acc1, gtknown_at_best_acc1,
                   best_loc1, acc1_at_best_loc1, gtknown_at_best_loc1, args)

        print("===========================================================")
        print("Start Evaluation on Best Checkpoint ...")

    args.resume = os.path.join(log_folder, 'model_best.pth.tar')
    model, _ = load_model(model, optimizer, args)
    evaluate(val_loader, model, normalize,  criterion, args , conv2d)
    cam_curve(val_loader, model, normalize, criterion, writer, args)
    

def hard_example_mining(dist_mat_p , dist_mat_n , labels, return_inds=False):

    N = dist_mat_p.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    
    dist_ap, relative_p_inds = torch.max(
        dist_mat_p[is_pos].contiguous().view(N, -1), 1, keepdim=True)
   
    dist_an, relative_n_inds = torch.min(
        dist_mat_n[is_pos].contiguous().view(N, -1), 1, keepdim=True)
 
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an



def cosine_dist(x, g1, g2):
    dot_p = x @ g1.t()
    norm1 = torch.norm(x, 2, 1) + 1e-8
    norm2 = torch.norm(g1, 2, 1)+ 1e-8
    dot_p = torch.div(dot_p, norm1.unsqueeze(1))
    dot_p = torch.div(dot_p, norm2)
    
    
    dot_n = x @ g2.t()
    norm3 = torch.norm(g2, 2, 1)+ 1e-8
    dot_n = torch.div(dot_n, norm1.unsqueeze(1))
    dot_n = torch.div(dot_n, norm3)
    
    return 1.0 - dot_p , 1.0 - dot_n


def exemplar_loss_fn(x , g1, g2 , y, a) :

    dist_mat_p , dist_mat_n = cosine_dist(x, g1, g2)
    dist_ap, dist_an = hard_example_mining(dist_mat_p , dist_mat_n, a, return_inds=False)
    y = dist_an.new().resize_as_(dist_an).fill_(1)
    loss = ranking_loss(dist_an - dist_ap, y)
    return loss

def train(train_loader, model, criterion, optimizer, epoch, args , attack , num_classes):
    batch_time = SumEpochMeter('Time', ':6.3f')
    data_time = SumEpochMeter('Data', ':6.3f')
    losses = AverageEpochMeter('Loss', ':.4e')
    top1 = AverageEpochMeter('Top-1 Classification Acc')
    top5 = AverageEpochMeter('Top-5 Classification Acc')
    learning_rate = AverageEpochMeter('Learning Rate:', ':.1e')
    progress = ProgressEpochMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate, top1, top5],
        prefix="\nTraining Phase: ")

    for param_group in optimizer.param_groups:
        learning_rate.update(param_group['lr'])
        break

    # switch to train mode
    model.train()
    end = time.time()

    means = [.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)
    
    mean = torch.tensor([.485, .456, .406])
    std = torch.tensor([.229, .224, .225])
    
    def normalize(x):
        new_std = std[..., None, None]
        new_mean = mean[..., None, None]
        return (x - new_mean.cuda())/new_std.cuda()
    
    avg_loss=0
    avg_loss_sal = 0.
        
    for i, (x,y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            y = Variable(y.cuda())
            x = Variable(x.cuda())
       

        optimizer.zero_grad()
        
        xadv = attack.TopK_sal_attack(model, x, y, 0., 1.) 
        
        x_ = normalize(torch.cat([x,xadv] , 0))
        batch_size = x_.size(0)
        y_ = y.repeat(2)
        a_ = torch.cat((torch.arange(0,x.size(0)), torch.arange(0,x.size(0))), 0).long()   
        a_ = Variable(a_).cuda() 
           
            
        x_.requires_grad = True                        
        scores , _  = model({0:x_,1:1})
                                  
        top_scores = scores.gather(1 , index = y_.unsqueeze(1))             
        non_target_indices= torch.from_numpy(np.array([[k for k in range(num_classes) if k != y_[j]] for j in range(y_.size(0))] )).cuda()        

        bottom_scores = scores.gather(1 , index = non_target_indices) 
        bottom_scores = bottom_scores.max(dim = 1)[0]
                
        g1 = torch.autograd.grad( top_scores.mean() , x_ , retain_graph=True,create_graph=True)[0]  
        g2 = torch.autograd.grad( bottom_scores.mean() , x_ , retain_graph=True,create_graph=True)[0]                
         
            
        g1_adp = torch.mean(torch.abs(g1),1)
        g2_adp = torch.mean(torch.abs(g2),1)
        x_conv = torch.mean(torch.abs(x_.detach()),1)
            
        x_.requires_grad = False
        exemplar_loss = exemplar_loss_fn(x_conv.reshape(batch_size,-1), g1_adp.reshape(batch_size,-1), g2_adp.reshape(batch_size,-1) , y_ , a_)
        scores_adv , _ = model({0:normalize(xadv),1:0})            
            
        loss = criterion(scores_adv, y)             
        optimizer.zero_grad()
        total_loss = loss + 0.6*exemplar_loss
        total_loss.backward()       
        optimizer.step()
            
        avg_loss = avg_loss+loss.data.item()
        avg_loss_sal = avg_loss_sal+exemplar_loss.data.item()
            
        if i % 10 ==0:
            print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Sal Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1) , avg_loss_sal/float(i+1)  ))
                

        # measure accuracy and record loss
        acc1, acc5 = accuracy(scores_adv, y, topk=(1, 5))
        losses.update(total_loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
    return top1.avg, losses.avg, progress


if __name__ == '__main__':
    main()
