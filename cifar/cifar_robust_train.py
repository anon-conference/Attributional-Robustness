import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import backbone
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
from pgd import *
from saliency_pgd import *
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision
from PIL import Image
import numpy as np

print("Process id ", os.getpid())
ranking_loss = nn.SoftMarginLoss()
cos = nn.CosineSimilarity(dim=1, eps=1e-15)

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

def saliency_train(base_loader, val_loader, model, start_epoch, stop_epoch, params, tmp):    
    print("saliency training started")
    config = {
            'epsilon': 8.0 / 255.0,
            'num_steps': 3,
            'step_size': 5./ 255.0,
            'k_top': 1000
            }
    
    print(config)
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    
    def normalize(x):
        new_std = std[..., None, None]
        new_mean = mean[..., None, None]
        return (x - new_mean.cuda())/new_std.cuda()
    
    attack = AttackSaliency(config , normalize)
    
    config1 = {
        'epsilon': 8.0 / 255.0,
        'num_steps': 40,
        'step_size': 2.0/ 255.0,
    }
    print("PGD ", config1)

    attack1 = AttackPGD(config1 , normalize) 
    
    lossfn = nn.CrossEntropyLoss()
    ranking_loss = nn.SoftMarginLoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    softmax_F = nn.Softmax(dim = 1)
    
    params_optimize = list(model.parameters())
    optimizer = torch.optim.SGD(params_optimize , lr=0.1 , nesterov = True , momentum = 0.9,weight_decay=2e-4)
    
    def lr_lambda(steps):
        if steps < 50:
            return 1.0
        elif steps in range(50,80):
            return 0.1
        elif steps in range(80,150):
            return 0.01
        elif steps in range(150,200):
            return 0.005
        else:
            return 0.005
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    params.save_freq = 10
    max_acc = 0 
    
    print("start_epoch", start_epoch, "stop_epoch" , stop_epoch)
    for initial_step in range(start_epoch):
        if initial_step == 0:
            print("restoring step size schudeler")
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()
        
    epoch = 0
    for epoch in range(start_epoch , stop_epoch):
        model.train()
        print_freq = 10
        avg_loss=0
        avg_loss_sal = 0.
        for i, (x , y) in enumerate(base_loader):
                                       
            y = Variable(y.cuda())
            x = Variable(x).cuda()
            
            optimizer.zero_grad()
            
            xadv = attack.topk_alignment_saliency_attack(model, x, y, 0., 1.)                           
            x_ = normalize(torch.cat([x,xadv] , 0))
            batch_size = x_.size(0)
            y_ = y.repeat(2)
            a_ = torch.cat((torch.arange(0,x.size(0)), torch.arange(0,x.size(0))), 0).long()   
            a_ = Variable(a_).cuda() 
           
            
            x_.requires_grad = True                        
            _ , scores  = model({0:x_,1:1})
                                  
            top_scores = scores.gather(1 , index = y_.unsqueeze(1))             
            non_target_indices= np.array([[k for k in range(10) if k != y_[j]] for j in range(y_.size(0))] )            
            bottom_scores = scores.gather(1 , index = torch.tensor(non_target_indices).cuda() ) 
            bottom_scores = bottom_scores.max(dim = 1)[0]
                
            g1 = torch.autograd.grad( top_scores.mean() , x_ , retain_graph=True,create_graph=True)[0]  
            g2 = torch.autograd.grad( bottom_scores.mean() , x_ , retain_graph=True,create_graph=True)[0]                

            g1_adp = g1
            g2_adp = g2
            x_conv = x_
            
            x_.requires_grad = False
            exemplar_loss = exemplar_loss_fn(x_conv.reshape(batch_size,-1), g1_adp.reshape(batch_size,-1), g2_adp.reshape(batch_size,-1)  , y_, a_)
            _ , scores_adv  = model({0:normalize(xadv),1:0})            
            
            loss = lossfn(scores_adv, y)             
            optimizer.zero_grad()
            total_loss = loss + 0.5*exemplar_loss
            total_loss.backward()       

            optimizer.step()
            
            avg_loss = avg_loss+loss.data.item()
            avg_loss_sal = avg_loss_sal+exemplar_loss.data.item()
            
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Sal Loss {:f}'.format(epoch, i, len(base_loader), avg_loss/float(i+1) , avg_loss_sal/float(i+1)  ))
           
        print("lr is ",optimizer.state_dict()['param_groups'][0]['lr'])
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        
        scheduler.step()
        model.eval()
        correct = correct_adv = total = 0.
        
        if epoch %1 ==0:
            for i,(x,y) in enumerate(val_loader):
                x = x.cuda()
                y = y.cuda()
                _ , scores = model.forward({0:normalize(x),1:0})
                if epoch %10 ==0:
                    optimizer.zero_grad()
                    xadv = attack1.attack(model,x , y , 0. , 1.)
                    _ , scores_adv = model.forward({0:normalize(xadv) , 1:0})
                    p1_adv = torch.argmax(scores_adv,1)
                    correct_adv += (p1_adv==y).sum().item()

                p1 = torch.argmax(scores,1)
                correct += (p1==y).sum().item()
                total += p1.size(0)

        print("Epoch {0} :  : Accuracy {1} : Adv Accuracy {2}".format(epoch, (float(correct)*100)/total , (float(correct_adv)*100)/total )) 

        if ((epoch % params.save_freq==0) or (epoch==stop_epoch-1)) :
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
        epoch += 1

    return model

params = parse_args('train')
params.model = 'WideResNet28_10_r'
params.dataset = 'cifar'
params.method = 'srt'

class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

    
class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        

    def __call__(self, inp):
        out1 = self.train_transform(inp)
        return out1

#Data Loading     
train_transform = TransformsC10() 

test_transform = train_transform.test_transform


trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8 )

testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=8)

beta_val = 50.0
model = backbone.WideResNet28_10_r( flatten = True, beta_value = beta_val)
params.checkpoint_dir = './checkpoints/%s/%s_%s_%s' %(params.dataset, params.model, params.method , 'cifar_saliency_b50')
        
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

start_epoch = params.start_epoch
stop_epoch = params.stop_epoch

tmp = {}
model = nn.DataParallel(model, device_ids= range(torch.cuda.device_count()))    
if params.resume:
    print("resuming" , params.checkpoint_dir)
    resume_file = get_resume_file(params.checkpoint_dir )
    if resume_file is not None:
        print("resume_file" , resume_file)
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])

        
print(params)
print("beta value in softplus:" , beta_val)   

model = model.cuda()

model = saliency_train(trainloader, testloader,  model, start_epoch, 201, params,tmp)

