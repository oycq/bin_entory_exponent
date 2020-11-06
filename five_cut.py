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
        ctx.save_for_backward(input)
        return (input >= 0).float() * 2 -1


    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input.abs() > 1)] = 0
        return grad_input 


class Net(nn.Module):
    def __init__(self, hid=400, out_features=10):
        super(Net, self).__init__()
        CLASS = 10
        selector = torch.zeros(64,7,7).long()
        for i in range(64):
            for j in range(7):
                alpha = (i / 8 * 3 + j) * 28 + (i % 8) * 3
                selector[i,j] = torch.arange(7).long() + alpha
        self.selector = selector.reshape(64,-1)

        K = (49 + 1) ** 0.5
        self.weight_1 = torch.randn((64, 49, hid)) / K
        self.bias_1   = torch.randn((64, 1, hid)) / K
        self.norm1 = nn.BatchNorm1d(64 * hid)

        K = (hid * 0.5 + 1) ** 0.5
        self.weight_2 = torch.randn((64, hid, hid)) / K
        self.bias_2   = torch.randn((64, 1, hid)) / K
        self.norm2 = nn.BatchNorm1d(64 * hid)

        self.weight_3 = torch.randn((64, hid, hid)) / K
        self.bias_3   = torch.randn((64, 1, hid)) / K
        self.norm3 = nn.BatchNorm1d(64* hid)

        self.weight_4 = torch.randn((64, hid, out_features)) / K
        self.bias_4   = torch.randn((64, 1, out_features)) / K

        self.weight_5 = torch.randn((64*out_features, 1, CLASS))

        self.score_K = torch.zeros(1) + 1

        self.weight_1 = torch.nn.Parameter(self.weight_1)
        self.weight_2 = torch.nn.Parameter(self.weight_2)
        self.weight_3 = torch.nn.Parameter(self.weight_3)
        self.weight_4 = torch.nn.Parameter(self.weight_4)
        self.weight_5 = torch.nn.Parameter(self.weight_5)
        self.bias_1= torch.nn.Parameter(self.bias_1)
        self.bias_2= torch.nn.Parameter(self.bias_2)
        self.bias_3= torch.nn.Parameter(self.bias_3)
        self.bias_4= torch.nn.Parameter(self.bias_4)
        self.score_K = torch.nn.Parameter(self.score_K)

    def _apply_batch_norm(self, x, norm_layer):
        base_size = x.shape[0]
        batch_size = x.shape[1]
        x = x.permute(1,0,2)
        x = x.reshape(batch_size, -1)
        x = norm_layer(x)
        x = x.reshape(batch_size, base_size, -1)
        x = x.permute(1,0,2)
        return x

    def forward(self, inputs,debug = 0,quant = 0):
        quantized = Quantized.apply
        relu = nn.ReLU(inplace = True)
        x = inputs[:, self.selector]
        x = x.permute(1,0,2)
        x = x.bmm(self.weight_1)
        x = x + self.bias_1
        x = self._apply_batch_norm(x, self.norm1)
        x = relu(x)

        x = x.bmm(self.weight_2)
        x = x + self.bias_2
        x = self._apply_batch_norm(x, self.norm2)
        x = relu(x)
       
        x = x.bmm(self.weight_3)
        x = x + self.bias_3
        x = self._apply_batch_norm(x, self.norm3)
        x = relu(x)

        x = x.bmm(self.weight_4)
        x = x + self.bias_4

        x = x.permute(0,2,1)
        x = x.reshape(-1, x.shape[2]).unsqueeze(-1)
        x = x.bmm(self.weight_5)
        apple = abs(x) - 1
        constraint = (apple * apple).mean().sqrt()
        z = x
        if quant:
            x = quantized(x)
     #   else:
     #       noise = torch.cuda.FloatTensor(x.shape).normal_() * constraint
     #       x = x + noise
        x = x.permute(1,2,0)
        y = x
        x = x.mean(-1) * self.score_K
        if debug:
            print(z[0,0,:])
            print(y[0,0,:])
            print(x[0]/self.score_K)
            print(self.score_K)
        return x,constraint

def get_loss_acc(x, labels):
    accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1).mean()
    return loss, accurate

net = Net(100, 20).cuda()
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
 
