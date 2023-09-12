#! -*- coding: utf-8 -*-
##特征提取后分类
import pylab as pl
import numpy as np
import pandas as pd

from astropy.table import Table
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import *
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os

batch_size_test = 200
batch_size_train = 100

test_no = 15

num_epoch = 150
log_intertrain = 1

torch.cuda.set_device(0)
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

class FeatureDataset(Dataset):
    def __init__(self,feature_file,label_file):
        self.features = np.array(pd.read_csv(feature_file))
        self.labels = np.array(pd.read_csv(label_file))      
        print("------读取文件结束------")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        feature = torch.Tensor(self.features[idx])
        label = torch.LongTensor(self.labels[idx])
        return feature,label

feature_dir = 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\'
label_dir = 'C:\\Users\\23650\\Desktop\\research\\NAO\\datasets\\'
trainData = FeatureDataset(feature_dir + '{}\\feature_valSet.csv'.format(test_no),label_dir + 'valSet.csv')
trainLoader = torch.utils.data.DataLoader(
    dataset=trainData, batch_size=batch_size_train, shuffle=True
)

testData = FeatureDataset(feature_dir + '{}\\feature_testSet.csv'.format(test_no),label_dir + 'testSet.csv')
testLoader = torch.utils.data.DataLoader(
    dataset=testData, batch_size=batch_size_test, shuffle=False
)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(128,128),
            nn.PReLU(), 
            nn.Linear(128,128),
            nn.PReLU(),
            nn.Linear(128,4),
            nn.LogSoftmax()      
        )
    def forward(self, x):
        x = self.dense(x)
        return x

def train(epoch):
    torch.backends.cudnn.benchmark = True
    print('start train')
    D.train()
    total = 0
    goal = 0
    for batch_idx, (data,label) in enumerate(trainLoader):
                
        data = data.cuda()
        label = label.cuda().view(-1)
               
        result = D(data)
        result = result.cuda()

        loss = criterion(result, label)
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()               
        #torch.cuda.empty_cache() 
        
        result = result.cpu()
        label = label.cpu()
                
        if (batch_idx+1) % log_intertrain == 0:
            print('Epoch[{}/{}],loss:{:.6f}'.format(
                epoch, num_epoch, loss.data.item()
            ))
            with torch.no_grad():
                train_losses.append(loss.data.item())
                train_counter.append(
                    (batch_idx*batch_size_train)/len(trainLoader.dataset) + (epoch)) 
                for num in range(0,len(result)):
                    total = total + 1
                    predict = result[num,:].numpy().argmax()
                    real = label[num]
                    if predict == real:
                        goal += 1
                train_acc.append(goal/total)
                #print(result[0,:])
                #print(label[0])                      
    print('end train') 


def test(epoch):
    print('start test')
    D.eval()
    total = 0
    goal = 0
    test_loss = 0
    mat = np.zeros([4,4])
    
    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(testLoader):
            #print(batch_idx)
            data = data.cuda()
            label = label.cuda().view(-1)
            result = D(data)
            label = label.cpu()
            result = result.cpu()
            loss = criterion(result, label)
            temp = loss.cpu()
            #test_loss += d_loss
            for num in range(0,len(result)):
                total = total + 1
                predict = result[num,:].numpy().argmax()
                real = label[num]
                mat[predict,real] = mat[predict,real] + 1
                if predict == real:
                    goal += 1
            
            test_loss += batch_size_test * temp
        
    test_loss /= len(testLoader.dataset)
    test_losses.append(test_loss)
    test_counter.append(epoch)
    #print(test_loss)
    #print(total)
    #print(goal)
    test_acc.append(goal/total)
    #print(mat)
    dataframe = pd.DataFrame(mat)
    dataframe.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\result{}.csv'.format(test_no,epoch),index=False,sep=',') 
    print('end test')
    
D = DNN().cuda()

optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
criterion = nn.NLLLoss()

train_losses = []
train_counter = []
test_losses_MSE = []
test_counter = []
train_acc = []
test_losses = []
test_acc = []

for epoch in range(num_epoch):   
    train(epoch)
    if (epoch) % 5==0: 
        torch.save(D.state_dict(), 'C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\DNN-{}'.format(test_no,epoch) +  '.pth')    
    test(epoch)    

mat = np.array(test_acc) 
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\test_acc.csv'.format(test_no),index=False,sep=',')

mat = np.array(train_acc)
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\train_acc.csv'.format(test_no),index=False,sep=',')

mat = np.array(test_losses) 
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\test_losses.csv'.format(test_no),index=False,sep=',')

mat = np.array(test_counter)
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\test_counter.csv'.format(test_no),index=False,sep=',')

mat = np.array(train_losses) 
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\train_losses.csv'.format(test_no),index=False,sep=',')

mat = np.array(train_counter)
df = pd.DataFrame(mat)
df.to_csv('C:\\Users\\23650\\Desktop\\research\\NAO\\semi\\{}\\train_counter.csv'.format(test_no),index=False,sep=',')