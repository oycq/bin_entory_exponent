import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
import os
from test_mnist_dataset import trainloader, testloader
import cv2
import numpy as np
import random 
torch.manual_seed(0)
random.seed()

IF_WANDB = 1
OFFLINE = 0
IF_SAVE = 0
SIX = 6
BATCH_SIZE = 100
WORKERS = 15
CLASS = 10
TESTING_LEN = 10000
LUT_RANDN_K = 0
CONNECT_RANDN_K = 0.01
SIGMOID_K = 1
K_SCORE = 5

if IF_WANDB:
    if OFFLINE:
        os.environ["WANDB_API_KEY"] = '72217df9e0f4d28c36a5727db964576fb2caac26'
        os.environ["WANDB_MODE"] = "dryrun"
    import wandb
    wandb.init(project = 'path')#, name = '.')


class LutLayer(nn.Module):
    def __init__(self, depth):
        super(LutLayer, self).__init__()
        lut = torch.zeros(depth, 2 ** SIX)
        p_q_2_lut_table = torch.zeros(SIX*2, 2**SIX)
        for i in range(2**SIX):
            bins = ('{0:0%db}'%SIX).format(i)
            for j in range(SIX):
                if bins[j] == '0':
                    p_q_2_lut_table[j+SIX][i] = 1
                else:
                    p_q_2_lut_table[j][i] = 1
            a = 1.0 / SIX * bins.count('0')
            if a == 0:
                a = 0.01
            if a == 1:
                a = 0.99
            lut[:, i] = math.log(a/(1-a))

        self.p_q_2_lut_table = p_q_2_lut_table.cuda()
        self.lut = torch.nn.Parameter(lut)

    def get_lut_infer(self, bn):
        lut_infer = torch.zeros_like(self.lut) 
        lut = self.lut
        if bn is not None:
            lut = lut - bn.running_mean.unsqueeze(-1)
            lut = lut / ((bn.running_var.unsqueeze(-1) + 1e-05) ** 0.5)
            #lut = lut * bn.weight.unsqueeze(-1)
            #lut = lut - bn.bias.unsqueeze(-1)
        lut_infer[lut > 0] = 1
        return lut_infer


    def forward(self, inputs, infer=False, bn=None):
        lut_infer = self.get_lut_infer(bn)
        eps = 1e-7
        p_q = inputs.unsqueeze(2).repeat(1,1,2,1)
        p_q[:,:,0,:] = 1 - p_q[:,:,0,:]
        p_q = torch.nn.functional.relu(p_q) + eps
        p_q = p_q.view(p_q.shape[0], p_q.shape[1], -1)
        p_q_log = p_q.log()
        lut_p = (p_q_log.matmul(self.p_q_2_lut_table)).exp()
        if infer:
            output = (lut_p * lut_infer).sum(-1)
        else:
            output = (lut_p * self.lut).sum(-1)
        return output


class MyCNN(nn.Module):
    def __init__(self, inputs_d, kernal_d, kernal_a, stride):
        super(MyCNN, self).__init__()
        self.stride = stride
        self.kernal_d = kernal_d
        connect_kernal = torch.randn(kernal_d*SIX, inputs_d, kernal_a, kernal_a) *CONNECT_RANDN_K
        self.connect_kernal = torch.nn.Parameter(connect_kernal)
        self.lut_layer = LutLayer(kernal_d)
        self.conv_bn = nn.BatchNorm2d(kernal_d, affine=False)
        self.appointed_connect = torch.zeros_like(self.connect_kernal).cuda()
        for i in range(kernal_d):
            r = random.randrange(0,inputs_d)
            self.appointed_connect[i*SIX,r,:,:] += 6
            #self.appointed_connect[i*SIX,r,:,:] += 0

    def get_infer_kernal(self):
        connect_kernal_shape = self.connect_kernal.shape
        connect_kernal = self.connect_kernal/CONNECT_RANDN_K + self.appointed_connect
        connect_kernal= connect_kernal.view(self.kernal_d*SIX, -1)
        max_idx = connect_kernal.argmax(-1)
        connect_kernal_infer = torch.zeros_like(connect_kernal).\
                scatter(1, max_idx.unsqueeze(1), 1.0)
        connect_kernal_infer = connect_kernal_infer.view(connect_kernal_shape)
        return connect_kernal_infer

    def forward(self, inputs, infer=False, quant=False):
        inputs = inputs.detach()
        connect_kernal_shape = self.connect_kernal.shape
        connect_kernal= (self.connect_kernal/CONNECT_RANDN_K+self.appointed_connect).exp()
        connect_kernal = connect_kernal.view(self.kernal_d*SIX, -1)
        connect_kernal = connect_kernal / connect_kernal.sum(-1).unsqueeze(-1)
        connect_kernal = connect_kernal.view(connect_kernal_shape)
        connect_kernal_infer = self.get_infer_kernal()
        if True:
        #if infer:
            x = F.conv2d(inputs.contiguous(), connect_kernal_infer, stride=self.stride)
        else:
            x = F.conv2d(inputs.contiguous(), connect_kernal, stride=self.stride)
        x = x.permute(0,2,3,1)
        output_shape = (x.shape[0],x.shape[1],x.shape[2],self.kernal_d)
        x = x.reshape(-1, self.kernal_d, SIX)
        x = self.lut_layer(x, infer, bn=self.conv_bn)
        x = x.view(output_shape).permute(0,3,1,2)
        if infer:
            return x, x
        x = self.conv_bn(x)
        x = x + torch.randn_like(x) * 0.2
        x_norm = x.clone()
        x = torch.sigmoid(x*SIGMOID_K)
        r = torch.cuda.FloatTensor(x.shape).uniform_()
        x_quant = (x > r).float()
        return x_norm, x_quant


class NetA(nn.Module):
    def __init__(self, input_size=784):
        super(NetA, self).__init__()
        self.cnn1 = MyCNN(inputs_d=1, kernal_d=32, kernal_a=8, stride = 4) #(1,1,1280)1280
        self.cnn2 = MyCNN(inputs_d=32, kernal_d=64, kernal_a=3, stride = 1) #(1,1,1280)1280
        self.cnn3 = MyCNN(inputs_d=64, kernal_d=1000, kernal_a=4, stride = 1) #(1,1,1280)1280

    def forward(self, inputs, infer=False):
        quant_state = True
        x1,x1_quant = self.cnn1(inputs , infer, quant=quant_state)
        x2,x2_quant = self.cnn2(x1_quant, infer, quant=quant_state)
        x3,x3_quant = self.cnn3(x2_quant, infer, quant=quant_state)
        xo = x3_quant
        return x1,x2,x3,xo

class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.C1 = torch.nn.Sequential( 
                        torch.nn.Conv2d(32 , 64, 3, stride=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(64, 128, 4, stride=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(128, 256, 1, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(256, 10, 1, stride=1),
                        )
        self.C2 = torch.nn.Sequential( 
                        torch.nn.Conv2d(64, 256, 4, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(256, 256, 1, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(256, 10, 1, stride=1),
                        )
        self.C3 = torch.nn.Sequential( 
                        torch.nn.Conv2d(1000, 256, 1, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(256, 256, 1, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace = True),
                        torch.nn.Conv2d(256, 10, 1, stride=1),
                        )

        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x1, x2, x3, xo, labels):
        s1 = self.C1(x1).squeeze(-1).squeeze(-1)
        l1 = self.loss_function(s1, labels)
        s2 = self.C2(x2).squeeze(-1).squeeze(-1)
        l2 = self.loss_function(s2, labels)
        s3 = self.C3(x3).squeeze(-1).squeeze(-1)
        l3 = self.loss_function(s3, labels)
        so  = (xo.view(xo.shape[0],-1).view(xo.shape[0], CLASS, -1).mean(-1) - 0.5) * K_SCORE
        lo = self.loss_function(so, labels)
        return l1,l2,l3,lo

def get_L2_loss():
    a = ((netA.cnn1.lut_layer.lut) ** 2).mean()
    b = ((netA.cnn2.lut_layer.lut) ** 2).mean()
    c = ((netA.cnn3.lut_layer.lut) ** 2).mean()
    return a+b+c

def get_loss_acc(x1,x2,x3,xo,labels):
    global DEBUG
    l1, l2, l3, lo = netC(x1, x2, x3, xo, labels)
    critic_loss = (lo.detach() - l3) ** 2 + (l3.detach() - l2) ** 2 + (l2.detach() - l1) ** 2
    critic_loss = critic_loss.mean()
    if DEBUG:
        #print('%100.4f %8.4f %8.4f'%(l1.mean(),l2.mean(),l3.mean()))
        DEBUG = 0
    actor_loss = l1 + l2 + l3
    actor_loss = actor_loss.mean()
    l2_loss = get_L2_loss()
    so  = (xo.view(xo.shape[0],-1).view(xo.shape[0], CLASS, -1).mean(-1) - 0.5) * K_SCORE
    accurate = (so.argmax(-1) == labels).float().mean() * 100
    return actor_loss, critic_loss, l2_loss, accurate

best = 0
def get_fpga_acc(train=True):
    global best
    total_correct = 0
    total_items = 0
    if train:
        loader = trainloader
    else:
        loader = testloader
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data[0].cuda(), data[1].cuda()
            x1,x2,x3,xo = netA(images,infer=True)
            _, _, _, accurate = get_loss_acc(x1,x2,x3,xo,labels)
            total_correct += accurate.item() * labels.shape[0]
            total_items += labels.shape[0]
            if total_items >= TESTING_LEN:
                break
        acc = total_correct / total_items
    if train:
        print('train_acc:%8.3f%%'%(acc))
        if IF_WANDB:
            wandb.log({'train_acc':acc})
    else:
        if best < acc:
            best = acc
        print(' test_acc:%8.3f%%   %8.3f%%'%(acc,best))
        if IF_WANDB:
            wandb.log({'test_acc':acc})

netA = NetA().cuda()
netC = NetC().cuda()
optimizerA = optim.Adam(netA.parameters())
optimizerC = optim.Adam(netC.parameters())

DEBUG = False
for epoch in range(1000000):
    print(epoch)
    SIGMOID_K = epoch / 600.0 + 1
    if epoch % 5 == 0:
        optimizerC = optim.Adam(netC.parameters())
    if epoch % 5 == 2:
        optimizerA = optim.Adam(netA.parameters())

    for i, data in enumerate(trainloader, 0):
        images,labels = data[0].cuda(), data[1].cuda()
        x1,x2,x3,xo = netA(images)
        actor_loss, critic_loss, l2_loss, acc = get_loss_acc(x1,x2,x3,xo,labels)
        if (epoch % 5) > 1:
        #if critic_loss < 0.05:
            loss = actor_loss + l2_loss * 0.000
            optimizerA.zero_grad()
            loss.backward()
            optimizerA.step()
        else:
            loss = critic_loss
            optimizerC.zero_grad()
            loss.backward()
            optimizerC.step()
        if i % 50 == 0:
            DEBUG = 1
            print('%5d     acc:%7.3f    actor:%7.4f     critic:%7.4f    l2_loss:%7.4f'\
                    %(i,acc,actor_loss, critic_loss,l2_loss))
            if IF_WANDB:
                wandb.log({'acc':acc})
    get_fpga_acc(train = True)
    get_fpga_acc(train = False)
    if epoch % 20 == 0 and IF_SAVE:
        torch.save(netA.state_dict(), './models/path_norm_2_%d.model'%epoch)
        #death(i)


