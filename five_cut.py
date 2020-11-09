import mnist_web
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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

IF_WANDB = 0
IF_SAVE = 0
LAYER_UNITS = 2000
LAYERS = 3 
CLASS = 10
BATCH_SIZE = 100
NAME = 'neural_400_100'
WORKERS = 15

if IF_WANDB:
    import wandb
    wandb.init(project = 'neural', name = NAME)

dataset = my_dataset.MyDataset(train = True, margin = 3, noise_rate = 0.05)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()

class Quantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        r = torch.cuda.FloatTensor(input.shape).uniform_()
        return (input >= r).float() * 2 -1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2

class BLayer(nn.Module):
    def __init__(self, in_features, out_features, hid):
        super(BLayer, self).__init__()
        mask = torch.randn(out_features, in_features)
        mask_u = torch.zeros(out_features, 1)
        mask_sigma = torch.zeros(out_features, 1) + 1
        self.mask = torch.nn.Parameter(mask)
        self.mask_u = torch.nn.Parameter(mask_u)
        self.mask_sigma = torch.nn.Parameter(mask_sigma)
        W1 = torch.randn((out_features, in_features, hid)) / 2.236
        W2 = torch.randn((out_features, hid, hid))/ (hid ** 0.5)
        W3 = torch.randn((out_features, hid, hid)) / (hid ** 0.5)
        W4 = torch.randn((out_features, hid, 1)) / (hid ** 0.5)
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(W2)
        self.W3 = torch.nn.Parameter(W3)
        self.W4 = torch.nn.Parameter(W4)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu= torch.nn.ReLU(inplace=True)
        self.quantized = Quantized.apply

    def _s(self,x):
        x = x.bmm(self.W1)
        x = self.relu(x)
        x = x.bmm(self.W2)
        x = self.relu(x)
        x = x.bmm(self.W3)
        x = self.relu(x)
        x = x.bmm(self.W4)
        return x


    def _quantized_mask(self, debug = 0):
        mask = self.mask
        mean = mask.mean(-1).unsqueeze(-1)
        std = mask.std(-1).unsqueeze(-1)
        mask = (mask - mean) / std
        mask = (mask * self.mask_sigma) + self.mask_u
        mask = self.sigmoid(mask)
        mask_loss = (mask.sum(-1) - 5)
        mask_loss = (mask_loss * mask_loss).mean(-1)
        mask = (self.quantized(mask) + 1) / 2
        if debug:
            print('%6.3f %6.3f %6.3f'%(self.mask_sigma[0], self.mask_u[0],mask.sum(-1).mean()))
        return mask, mask_loss

    def forward(self, inputs, debug = 0):
        #inputs  : [batch, in_features] -> [in_features, batch, 1]
        inputs = inputs.t().unsqueeze(-1)
        mask, mask_loss = self._quantized_mask(debug)
        #mask    : [out_features, in_features] -> [in_features, 1, out_features]
        mask = self.mask.t().unsqueeze(1)
        x = inputs.bmm(mask)
        #x       : [in_features, batch, out_features] -> [out_features, batch, in_features]
        x = x.permute(2,1,0)
        x = self._s(x)
        #x       : [out_features, batch, 1] -> [batch, out_features]
        x = self.sigmoid(x)
        x = x.squeeze(-1).t()
        #x = self.quantized(x)
        return x, mask_loss

class Net(nn.Module):
    def __init__(self, hid=100, f=[800,600,400],out_features=100):
        super(Net, self).__init__()
        self.b0 = BLayer(784,f[0], hid)
        self.b1 = BLayer(f[0],f[1], hid)
        self.b2 = BLayer(f[1],f[2], hid)
        self.b3 = BLayer(f[2],out_features * 10, hid)
        self.score_K = torch.zeros(1) + 1
        self.score_K = torch.nn.Parameter(self.score_K)


    def forward(self, inputs, debug = 0):
        x, l1 = self.b0(inputs,debug)
        x, l2 = self.b1(x,debug)
        x, l3 = self.b2(x,debug)
        x, l4 = self.b3(x,debug)
        x = x.reshape(x.shape[0],10,-1)
        x = x.mean(-1) * self.score_K
        if debug:
            print(x[0])
            print('k %5.2f'%(self.score_K))
        return x, (l1+l2+l3+l4)/4


def get_loss_acc(x, labels):
    accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1).mean()
    return loss, accurate

net = Net(100, [800,600,400],200).cuda()
optimizer = optim.Adam(net.parameters())
for i in range(100000):
    debug = 0
    if i % 10 == 0:
        debug = 1
    images, labels = data_feeder.feed()
    x, mask_loss = net(images,debug)
    loss, accurate = get_loss_acc(x, labels)
    if i > 2000:
        loss += mask_loss * 0.01
    if debug:
        print('%d %10.3f %10.3f %10.3f\n'%(i, loss, accurate, mask_loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
