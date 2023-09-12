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

# Parameters
#batch_size_train = 128
batch_size_test = 1
img_size = 64
num_epochs = 50
log_interval = 10
momentum = 0.999
temperature = 0.07
test_no = 15

torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

# Define an image augmentation function
augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
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
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)

    def forward(self, x):
        return self.model(x)

class MoCo(nn.Module):
    def __init__(self, base_encoder, queue_size=8192):
        super(MoCo, self).__init__()

        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        for param_q in self.encoder_q.parameters():
            param_q.requires_grad = True
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(128, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size  # Store queue_size as an attribute
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Ensure the queue has been initialized
        if self.queue is None:
            self.queue = torch.zeros([self.feature_dim, self.queue_size], dtype=keys.dtype, device=keys.device)

        # Replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.t()[:, :min(batch_size, self.queue_size - ptr)]
        
        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, x, x_aug):
        q = self.encoder_q(x)  # queries: NxC
        features = q.detach()  # Save a copy of features before normalization
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(x_aug)  # keys: NxC
            k = F.normalize(k, dim=1)

        self._dequeue_and_enqueue(k)
        
        return features, q, k  # Return features along with q and k

def Test(loader, name):
    print('start test')   
    result_array = []
    with torch.no_grad():
        print('start test')
        for batch_idx, (img, img_aug) in enumerate(loader):
            img = img.cuda()
            img_aug = img_aug.cuda()
            results, _, _ = model(img, img_aug)
            for i in range(batch_size_test):
                result = results.view(-1,128).cpu().detach()[i].numpy()  # Save all features
                result_array.append(result)  # Use append instead of extend

        df = pd.DataFrame(result_array)
        df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\feature_{}.csv'.format(test_no, name), index=False, sep=',')   
                                    
    print('end test') 

model = MoCo(ResNet()).cuda()

# Load the model weights
model_path = 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\ResNet-145.pth'.format(test_no)  # Replace with your path
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

valData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\valSet.hdf5')
valLoader = torch.utils.data.DataLoader(
    dataset=valData, batch_size=batch_size_test, shuffle=False
)

testData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\testSet.hdf5')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)

trainData = ImgDataset('C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\trainSet.hdf5')
trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=batch_size_test, shuffle=False
)

Test(valLoader, 'valSet')
Test(testLoader, 'testSet')
Test(trainLoader, 'trainSet')