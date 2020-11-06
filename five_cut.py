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
BATCH_SIZE = 3000
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


class Net(nn.Module):
    def __init__(self, hid=200, features=[800,600,400]):
        super(Net, self).__init__()
        self.s1 = nn.Sequential(
            nn.Linear(784, hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, features[0]),
            torch.nn.Sigmoid()
        )
        self.s2 = nn.Sequential(
            nn.Linear(features[0], hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, features[1]),
            torch.nn.Sigmoid()
        )
        self.s3 = nn.Sequential(
            nn.Linear(features[1], hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, hid),
            nn.ReLU(inplace = True),
            nn.Linear(hid, features[2] * CLASS),
            torch.nn.Sigmoid()
        )
        self.score_K = torch.zeros(1) + 1
        self.score_K = torch.nn.Parameter(self.score_K)


    def forward(self, inputs,debug = 0,quant = 0):
        quantized = Quantized.apply
        x = self.s1(inputs)
        x = quantized(x)
        x = self.s2(x)
        x = quantized(x)
        x = self.s3(x)
        x = quantized(x)
        x = x.reshape(x.shape[0],10,-1)

        x = x.mean(-1) * self.score_K
        if debug:
            print(x[0])
            print('k %5.2f'%(self.score_K))
        return x,0

def get_loss_acc(x, labels):
    accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1).mean()
    return loss, accurate

net = Net(400).cuda()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.Adagrad(net.parameters())
for i in range(30000):
    images, labels = data_feeder.feed()
    x,constraint = net(images)
    loss, accurate = get_loss_acc(x, labels)
    loss += constraint * constraint
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(i)
        x,_= net(images,1,quant=0)
        print('constraint %6.4f'%constraint)
        print('trainq0  %8.4f  %8.3f%%'%(loss, accurate))
        x,_= net(images,quant=1)
        loss, accurate = get_loss_acc(x, labels)
        print('trainq1  %8.4f  %8.3f%%'%(loss, accurate))
        x,_ = net(images_t[:5000],quant=1)
        loss_t, accurate_t = get_loss_acc(x, labels_t[:5000])
        print('testq1   %8.4f  %8.3f%%'%(loss_t, accurate_t))
        x,_ = net(images_t[:5000],quant=0)
        loss_t, accurate_t = get_loss_acc(x, labels_t[:5000])
        print('testq0   %8.4f  %8.3f%%'%(loss_t, accurate_t))

        print('')
 
