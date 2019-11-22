import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
from scipy import stats

class AttackSaliency(nn.Module):
    def __init__(self,  config , normalize, n_classes=10):
        super(AttackSaliency, self).__init__()
        self.step_size = config['step_size']
        self.eps = config['epsilon']
        self.num_steps = config['num_steps']
        self.k_top = config['k_top']
        self.n_class = n_classes
        self.criterion = F.cross_entropy
        self.normalize_fn = normalize
        self.im_size = 32
        self.target_map = None        
        self.num_ig_steps = 100
        self.c = 3                   
        self.ranking_loss = nn.SoftMarginLoss()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        

    def topk_alignment_saliency_attack(self, model, images, y, min_value, max_value):
        
        def exemplar_loss_fn(x , g1, g2 , y) :
            dist_ap = 1.0 - self.cos(x,g1)
            dist_an = 1.0 - self.cos(x,g2)
            y = dist_an.new().resize_as_(dist_an).fill_(1)
            loss = self.ranking_loss(dist_an - dist_ap, y)
            return loss
        
        batch_size = images.size(0)
        adv = images.clone()
        with torch.no_grad():
            adv = adv + 2* self.eps * (torch.rand(images.size()).cuda() - 0.5)
            adv = torch.clamp(adv , min_value , max_value)
        
        adv.requires_grad = True      
        
        _, logits = model.forward({0:self.normalize_fn(adv) , 1: 0}) #1:0 denotes relu and 1:1 denotes softplus
        top_scores = logits.gather(1 , index = y.unsqueeze(1))
        non_target_indices= np.array([[k for k in range(self.n_class) if k != y[j]] for j in range(y.size(0))] )
            
        bottom_scores = logits.gather(1 , index = torch.tensor(non_target_indices).cuda() ) 
        bottom_scores = bottom_scores.max(dim = 1)[0]
                
        g1 = torch.autograd.grad(top_scores.mean() , adv , retain_graph=True)[0]
        
        g1_adp = g1
        g1_adp = g1_adp.reshape(batch_size, -1)
            
        g1_abs = torch.abs(g1_adp)
            
        _ , top_idx_g1 = torch.topk(g1_abs, self.k_top ,1)
            
        
        for i in range(self.num_steps):
            adv.requires_grad = True
            _, logits_with_softplus = model.forward({0:self.normalize_fn(adv) , 1: 1}) #1:0 denotes relu and 1:1 denotes softplus
            top_scores = logits_with_softplus.gather(1 , index = y.unsqueeze(1))
            non_target_indices= np.array([[k for k in range(self.n_class) if k != y[j]] for j in range(y.size(0))] )
            
            bottom_scores = logits_with_softplus.gather(1 , index = torch.tensor(non_target_indices).cuda() ) 
            bottom_scores = bottom_scores.max(dim = 1)[0]
                
            g1 = torch.autograd.grad(top_scores.mean() , adv , retain_graph=True,create_graph=True)[0]  
            g2 = torch.autograd.grad(bottom_scores.mean() , adv , retain_graph=True,create_graph=True)[0]
            
            g1_adp = g1
            g2_adp = g2
            x_conv = self.normalize_fn(adv)
            
            g1_adp = g1_adp.reshape(batch_size, -1)
            g2_adp = g2_adp.reshape(batch_size, -1)
            
            top_g1 = g1_adp.gather(1 , index = top_idx_g1)
            top_g2 = g2_adp.gather(1 , index = top_idx_g1)
            
            adv_flatten = x_conv.reshape(batch_size, -1)    
            top_adv = adv_flatten.gather(1, index = top_idx_g1)

            exemplar_loss = exemplar_loss_fn(top_adv, top_g1, top_g2, y)
            topK_direction = torch.autograd.grad(exemplar_loss, adv)[0]            

            with torch.no_grad():
                adv = adv + self.step_size * torch.sign(topK_direction)
                adv = torch.min(adv, images+self.eps)
                adv = torch.max(adv, images-self.eps)
                adv = torch.clamp(adv, min_value,  max_value)        
        
        return adv 
    

    def topk_alignment_mean_saliency_attack(self, model, images, y, min_value, max_value):
        
        def exemplar_loss_fn(x , g1, g2 , y) :
            dist_ap = 1.0 - self.cos(x,g1)
            dist_an = 1.0 - self.cos(x,g2)
            y = dist_an.new().resize_as_(dist_an).fill_(1)
            loss = self.ranking_loss(dist_an - dist_ap, y)
            return loss
        
        batch_size = images.size(0)
        adv = images.clone()
        with torch.no_grad():
            adv = adv + 2* self.eps * (torch.rand(images.size()).cuda() - 0.5)
            adv = torch.clamp(adv , min_value , max_value)
        
        adv.requires_grad = True      
        
        logits , _= model.forward({0:self.normalize_fn(adv) , 1: 0}) #1:0 denotes relu and 1:1 denotes softplus
        top_scores = logits.gather(1 , index = y.unsqueeze(1))
        
        g1 = torch.autograd.grad(top_scores.mean() , adv , retain_graph=True)[0]
    
        g1_adp = torch.mean(torch.abs(g1),1).reshape(batch_size, -1)
            
        _ , top_idx_g1 = torch.topk(g1_adp, self.k_top ,1)
            
        
        for i in range(self.num_steps):
            adv.requires_grad = True
            logits_with_softplus , _ = model.forward({0:self.normalize_fn(adv) , 1: 1}) #1:0 denotes relu and 1:1 denotes softplus
            top_scores = logits_with_softplus.gather(1 , index = y.unsqueeze(1))
            non_target_indices= torch.from_numpy(np.array([[k for k in range(self.n_class) if k != y[j]] for j in range(y.size(0))] )).cuda()
#             print(non_target_indices)
            bottom_scores = logits_with_softplus.gather(1 , index = non_target_indices) 
            bottom_scores = bottom_scores.max(dim = 1)[0]
                
            g1 = torch.autograd.grad(top_scores.mean() , adv , retain_graph=True,create_graph=True)[0]  
            g2 = torch.autograd.grad(bottom_scores.mean() , adv , retain_graph=True,create_graph=True)[0]
            
            x_conv = torch.mean(torch.abs(self.normalize_fn(adv)),1)
            
            g1_adp = torch.mean(torch.abs(g1),1).reshape(batch_size, -1)
            g2_adp = torch.mean(torch.abs(g2),1).reshape(batch_size, -1)
            
            top_g1 = g1_adp.gather(1 , index = top_idx_g1)
            top_g2 = g2_adp.gather(1 , index = top_idx_g1)
            
            adv_flatten = x_conv.reshape(batch_size, -1)    
            top_adv = adv_flatten.gather(1, index = top_idx_g1)

            exemplar_loss = exemplar_loss_fn(top_adv, top_g1, top_g2, y)
            topK_direction = torch.autograd.grad(exemplar_loss, adv)[0]            

            with torch.no_grad():
                adv = adv + self.step_size * torch.sign(topK_direction)
                adv = torch.min(adv, images+self.eps)
                adv = torch.max(adv, images-self.eps)
                adv = torch.clamp(adv, min_value,  max_value)        
        
        return adv
    