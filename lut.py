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
import cv2

if IF_WANDB:
    import wandb
    wandb.init(project = 'lut_hard')#, name = '.')


dataset = my_dataset.MyDataset(train = True, margin = 2, noise_rate = 0.01)
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

class Quantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        r = torch.cuda.FloatTensor(input.shape).uniform_()
        return (input >= r).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


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

class ConnectLayer(nn.Module):
    def __init__(self, input_r=28, input_depth = 1, kernel_size=7, stride=3, output_depth=4):
        super(ConnectLayer, self).__init__()
        self.input_r = input_r
        self.input_depth = input_depth
        self.stride = stride
        self.kernel_size = kernel_size
        output_r = (input_r - kernel_size) * 1.0 / stride + 1
        if not output_r.is_integer():
            print('stride error')
            sys.exit(0)
        output_r = int(output_r)
        connect_w = torch.zeros((output_r**2)*output_depth*SIX, (input_r**2)*input_depth)
        connect_mask = torch.zeros_like(connect_w)
        for i in range(output_r):
            for j in range(output_r):
                ij_mask = self.get_ij_mask(i,j)
                k_base = (i * output_r + j) * output_depth * SIX
                for k in range(output_depth*SIX):
                    connect_mask[k_base + k] = ij_mask
        self.connect_mask = connect_mask.cuda()
        self.connect_w = torch.nn.Parameter(connect_w)

    def get_ij_mask(self,i,j):
        input_r = self.input_r
        input_depth = self.input_depth
        kernel_size = self.kernel_size
        stride = self.stride
        ij_mask = torch.zeros(input_r, input_r, input_depth)
        x_start, x_end = i * stride, i * stride + kernel_size
        y_start, y_end = j * stride, j * stride + kernel_size
        ij_mask[x_start : x_end, y_start : y_end, :] = 1
        ij_mask = ij_mask.view(-1)
        return ij_mask

    
    def forward(self, x):
        connect_w = self.connect_w
        connect_w = connect_w.exp()
        connect_w = connect_w * self.connect_mask
        connect_w = connect_w / connect_w.sum(-1).unsqueeze(-1)
        x = x.mm(connect_w.t())
        x = x.view(x.shape[0],-1,SIX)
        return x

    def infer(self, x):
        connect_w = self.connect_w
        max_idx = connect_w.argmax(-1)
        connect_w = torch.zeros_like(connect_w).scatter(1, max_idx.unsqueeze(1), 1.0)
        x = x.mm(connect_w.t())
        x = x.view(x.shape[0],-1,SIX)
        return x

    def visual(self, x):
        connect_w = self.connect_w
        max_idx = connect_w.argmax(-1)
        connect_w = torch.zeros_like(connect_w).scatter(1, max_idx.unsqueeze(1), 1.0)
        i = 0
        while(1):
            print(i)
            s = 0
            for j in range(6):
                a = connect_w[i*6+j]
                a = a.view(28,28).cpu().numpy()
                s = s + a
                cv2.imshow('%d'%j, cv2.resize(a,(640,640),interpolation = cv2.INTER_AREA))
            s[s>1] = 1
            cv2.imshow('s', cv2.resize(s,(640,640),interpolation = cv2.INTER_AREA))
            key = cv2.waitKey(0)
            if key == ord('q'):
                sys.exit(0)
                break
            if key == ord('-'):
                i += 1
            if key == ord('='):
                i -= 1
        sys.exit(0)


class CNNLayer(nn.Module):
    def __init__(self, input_r, input_depth, kernel_size, stride, output_depth):
        super(CNNLayer, self).__init__()
        self.lut_layer = LutLayer.apply
        output_r = (input_r - kernel_size) / stride + 1
        output_len = output_r * output_r * output_depth
        lut = torch.zeros(output_len, 2 ** SIX)
        self.lut = torch.nn.Parameter(lut)
        self.conect_layer1 = ConnectLayer(input_r,input_depth,kernel_size,stride,output_depth)
        self.norm = nn.BatchNorm1d(output_len)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        #x = self.conect_layer1.infer(x)
        x = self.conect_layer1(x)
        x = self.lut_layer(x, self.lut)
        x = self.norm(x)
        x = self.sigmoid(x)
        return x

    def infer(self, x, fixed_connect = True):
        r_std = (self.norm.running_var.unsqueeze(-1)) ** 0.5
        r_mean = self.norm.running_mean.unsqueeze(-1)
        k = self.norm.weight.unsqueeze(-1)
        b = self.norm.bias.unsqueeze(-1)
        lut_norm = ((self.lut - r_mean)/r_std)*k+b
        lut_infer = torch.zeros_like(self.lut)
        lut_infer[lut_norm > 0] = 1

        if fixed_connect:
            x = self.conect_layer1.infer(x)
        else:
            x = self.conect_layer1(x)
        x = self.lut_layer(x, lut_infer)
        return x


class Net(nn.Module):
    def __init__(self, input_size=784):
        super(Net, self).__init__()
        self.cnn1 = CNNLayer(28,1,6,2,8)
        self.cnn2 = CNNLayer(12,8,6,1,16)
        self.cnn3 = CNNLayer(7,16,4,1,64)
        self.cnn4 = CNNLayer(4,64,4,1,500)
        self.quantized = Quantized.apply
        score_K = torch.zeros(1) + 3
        self.score_K = torch.nn.Parameter(score_K)

    def forward(self, inputs):
        x = ((inputs + 1))/2
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.quantized(x)
        x = (x - 0.5) * self.score_K
        return x

    def infer(self, inputs, fixed_connect=True):
        with torch.no_grad():
            x = ((inputs + 1))/2
            x = self.cnn1.infer(x,fixed_connect)
            x = self.cnn2.infer(x,fixed_connect)
            x = self.cnn3.infer(x,fixed_connect)
            x = self.cnn4.infer(x,fixed_connect)
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

def get_test_acc(fixed_connect=True):
    acc = 0
    with torch.no_grad():
        for i in range(50):
            a = i * 200 
            b = i * 200 + 200
            images, labels = images_t[a:b], labels_t[a:b]
            x = net.infer(images,fixed_connect)
            loss, accurate = get_loss_acc(x, labels)
            acc += accurate.item() * 0.02
    if fixed_connect:
        print('test_fix:%8.3f%%'%acc)
        if IF_WANDB:
            wandb.log({'acc_test_fix':acc})
    else:
        print('test_flex:%8.3f%%'%acc)
        if IF_WANDB:
            wandb.log({'acc_test_flex':acc})




net = Net().cuda()
optimizer = optim.Adam(net.parameters())
#net.load_state_dict(torch.load('./lut_cnn_quant.model'))
#get_test_acc()

for i in range(60000000):
    images, labels = data_feeder.feed()
    x = net(images)
    loss,acc = get_loss_acc(x,labels)
    optimizer.zero_grad()
    loss.backward()
    if i % 50 == 0:
        print('%5d  %7.3f  %7.4f'%(i,acc,loss))
        if IF_WANDB:
            wandb.log({'acc':acc})
    if i % 400 == 0:
        get_test_acc(fixed_connect=True)
        get_test_acc(fixed_connect=False)
        print(net.score_K.item())
    if IF_SAVE and i % 10000 == 0:
        torch.save(net.state_dict(), 'lut_hard.model')

    optimizer.step()

