import numpy as np
import random
import sys
from cradle import Cradle
import torch
import time
from tqdm import tqdm
import my_dataset 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import cv2

IF_WANDB = 1
IF_SAVE = 1
SIX = 6
BATCH_SIZE = 1000
WORKERS = 15
CLASS = 10

if IF_WANDB:
    import wandb
    wandb.init(project = 'lut')#, name = '.')


dataset = my_dataset.MyDataset(train = True, margin = 0, noise_rate = 0)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()

delta_map = torch.zeros(2 ** SIX, SIX,2).long().cuda()
for i in range(2 ** SIX):
    bins = ('{0:0%db}'%SIX).format(i)
    for j in range(SIX):
        k = 2 ** j
        if bins[SIX-1-j] == '1':
            low = - k
            high = 0
        else:
            low = 0
            high = k
        delta_map[i,j,0] = low
        delta_map[i,j,1] = high

class LutLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, lut):
        #inputs  : [batch, features, SIX] 
        r = torch.cuda.FloatTensor(inputs.shape).uniform_()
        #r       : [batch, features, SIX] 
        inputs = (inputs>= r).long()
        #inputs  : [batch, features, SIX] 
        k1 = 2 ** torch.arange(0,SIX).cuda()
        #k1      : [SIX,]
        k2 = torch.arange(0, inputs.shape[1]).cuda() * (2 ** SIX)
        #k2      : [output_features,]
        lut_idx = inputs.long() *  k1
        #lut_idx : [batch, features, SIX]
        lut_idx = lut_idx.sum(-1) + k2
        #lut_idx : [batch, features]
        lut = lut.view(-1).clone()
        #lut     : [features * ( 2 ** SIX), ]
        output = lut[lut_idx]
        ctx.save_for_backward(inputs, lut, lut_idx)
        return output
        #out     : [batch, features]

    @staticmethod
    def backward(ctx, grad_father):
        #grad_father   : [batch, features]
        inputs, lut, lut_idx = ctx.saved_tensors
        #lut           : [features * ( 2 ** SIX), ]
        lut_grad = torch.zeros_like(lut)
        lut_grad = lut_grad.unsqueeze(0).repeat(inputs.shape[0],1)
        #lut_grad      : [batch, features * ( 2 ** SIX)]
        lut_grad.scatter_(-1, lut_idx, grad_father)
        lut_grad = lut_grad.sum(0)
        #lut_grad      : [features * ( 2 ** SIX)]
        lut_grad = lut_grad.view(-1,2**SIX)
        #lut_grad      : [features , ( 2 ** SIX)]

        delta_map_idx = lut_idx % (2 ** SIX)
        #delta_map_idx : [batch, features]
        delta = delta_map[delta_map_idx,:,:]
        #delta         : [batch, features, SIX, 2]
        grad_pair_idx = delta + lut_idx.unsqueeze(-1).unsqueeze(-1)
        #grad_pair_idx : [batch, features, SIX, 2]
        grad_pair = lut[grad_pair_idx]
        #grad_pair     : [batch, features, SIX, 2]
        input_grad = grad_pair[:,:,:,1] - grad_pair[:,:,:,0]
        input_grad = input_grad * grad_father.unsqueeze(-1)
        return input_grad, lut_grad
        #input_grad    : [batch, features, SIX]
        #lut_grad      : [features , ( 2 ** SIX)]


class Net(nn.Module):
    def __init__(self, f=[240,240,240], input_size=784):
        super(Net, self).__init__()
        self.lut_layer = LutLayer.apply
        lut1 = torch.zeros((f[0], 2 ** SIX))
        lut2 = torch.zeros((f[1], 2 ** SIX))
        lut3 = torch.zeros((f[2], 2 ** SIX))
        self.lut1= torch.nn.Parameter(lut1)
        self.lut2= torch.nn.Parameter(lut2)
        self.lut3= torch.nn.Parameter(lut3)
        self.connect_1 = torch.randint(input_size, (f[0], SIX))
        self.connect_2 = torch.randint(f[0], (f[1], SIX))
        self.connect_3 = torch.randint(f[1], (f[2], SIX))
        self.norm1 = nn.BatchNorm1d(f[0])
        self.norm2 = nn.BatchNorm1d(f[1])
        self.norm3 = nn.BatchNorm1d(f[2])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs, debug = 0, test=0):
        x = ((inputs + 1))/2
        x = x[:, self.connect_1]
        x = self.lut_layer(x, self.lut1)
#        x = self.norm1(x)
        x = self.sigmoid(x)
        x = x[:, self.connect_2]
        x = self.lut_layer(x, self.lut2)
#        x = self.norm2(x)
        x = self.sigmoid(x)
        x = x[:, self.connect_3]
        x = self.lut_layer(x, self.lut3)
        #x = self.sigmoid(x)
        return x

def get_loss_acc(x, labels):
    x = x.view(x.shape[0], CLASS, -1)
    x = x.mean(-1)
    accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1).mean()
    return loss, accurate

def get_test_acc():
    acc = 0
    with torch.no_grad():
        for i in range(50):
            a = i * 200 
            b = i * 200 + 200
            images, labels = images_t[a:b], labels_t[a:b]
            x = net(images)
            loss, accurate = get_loss_acc(x, labels)
            acc += accurate.item() * 0.02
    print('test:%8.3f%%'%acc)
    if IF_WANDB:
        wandb.log({'acc_test':acc})




net = Net([240,240,240],784).cuda()
optimizer = optim.Adam(net.parameters())

for i in range(3000000):
    images, labels = data_feeder.feed()
    x = net(images)
    loss,acc = get_loss_acc(x,labels)
    optimizer.zero_grad()
    loss.backward()
    if i % 50 == 0:
        print('%5d  %7.3f  %7.4f'%(i,acc,loss))
        if IF_WANDB:
            wandb.log({'acc':acc})
    if i % 300 == 0:
        get_test_acc()
    optimizer.step()

