#! -*- coding: utf-8 -*-
##特征提取
import random
from PIL import ImageFilter 
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
import torch.nn.functional as F
from torchvision import transforms

from torchvision.models import resnet18

import copy
import os

batch_size_test = 256
batch_size_train = 256
img_size = 64

log_interval = 10
momentum = 0.999
temperature = 0.07

test_no = 17

num_epoch = 150
log_interval = 10

torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

class ImgDataset(Dataset):
    def __init__(self,hdf5_file):
        data = Table.read(hdf5_file,path="/data")
        
        self.imgs = []
        
        print("------开始读取数据------")   
        for i in range(len(data)):
    
            #-------------------------------------------------
            # load in fimgs
            #
            img_r = data['r'][i]
            img_g = data['g'][i]
            img_b = data['b'][i]
            
            #-------------------------------------------------
            # rescale and clip fimgs according to the above 
            # scale factor and thresholds
            #          
            
            img_r_rscl = img_r + 0.5
            img_g_rscl = img_g + 0.5
            img_b_rscl = img_b + 0.5
            
            #-------------------------------------------------
            # determine scale factors and thresholds
            #      
                  
            img = np.zeros([3,64,64])
            img[0,:,:] = img_r_rscl
            img[1,:,:] = img_g_rscl
            img[2,:,:] = img_b_rscl

            
            self.imgs.append(torch.Tensor(img))
        
        print("------读取文件结束------")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].clone().detach()  # Clone and detach
        img_aug = augmentation(img.clone().detach())  # Clone and detach before augmenting
        return img, img_aug

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)

    def forward(self, x):
        return self.model(x)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


trainData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\trainSet.hdf5')
trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=batch_size_train, shuffle=True
)

trainSetLen = trainData.__len__()

testData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\testSet.hdf5')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)
        
model = MoCo(ResNet()).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_counter = []
test_losses = []
test_counter = []

# Training function
def train(epoch):
    print('start train')
        
    for batch_idx, (img, img_aug) in enumerate(trainLoader):
        img = img.cuda()  
        img_aug = img_aug.cuda()  
        output, target = model(img, img_aug)
        loss = criterion(output, target)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()      
        if (batch_idx + 1) % log_interval == 0:          
        
            print('Epoch[{}/{}],loss:{:.6f}'.format(
                epoch, num_epoch,loss.data.item()
            ))
            train_losses.append(loss.data.item())
            train_counter.append(
                #(batch_idx*batch_size_train)/len(train_loader.dataset) + (epoch))  
                (batch_idx*batch_size_train)/trainSetLen + (epoch))         
        
    print('end train') 

# Testing function
def test(epoch):
    print('start test')
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (img, img_aug) in enumerate(testLoader):
            img = img.cuda()  # Convert list to tensor and move to device
            img_aug = img_aug.cuda()  # Convert list to tensor and move to device
            output, target = model(img, img_aug)
            loss = criterion(output, target)
            test_loss += loss.data.item()

        test_loss /= len(testLoader.dataset)
        print('loss:{:.6f}'.format(test_loss))
        test_losses.append(test_loss)
        test_counter.append(epoch)
        
    print('end test')

# Training and Testing
for epoch in range(num_epoch):   
    train(epoch)
    if (epoch) % 5==0: 
        torch.save(model.state_dict(), 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet-{}'.format(test_no,epoch) +  '.pth')    
    test(epoch)    
    
    mat = np.array(train_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet_train_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(train_losses)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet_train_losses.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_counter)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet_test_counter.csv'.format(test_no),index=False,sep=',')   
    mat = np.array(test_losses)
    df = pd.DataFrame(mat)
    df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet_test_losses.csv'.format(test_no),index=False,sep=',') 