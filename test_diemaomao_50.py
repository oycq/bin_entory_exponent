import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
from test_mnist_dataset import trainloader, testloader
import cv2
import numpy as np
import random 
torch.manual_seed(0)
random.seed()

IF_WANDB = 1
IF_SAVE = 0
SIX = 6
BATCH_SIZE = 100
WORKERS = 15
CLASS = 10
TESTING_LEN = 10000
LUT_RANDN_K = 0
CONNECT_RANDN_K = 0.01
LUT_SCALE = 50
K_SCORE = 20

if IF_WANDB:
    import wandb
    wandb.init(project = 'ac')#, name = '.')



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
            lut[:, i] = math.log(a/(1-a)) / LUT_SCALE

        self.p_q_2_lut_table = p_q_2_lut_table.cuda()
        self.lut = torch.nn.Parameter(lut)

    def forward(self, inputs, infer=False):
        lut_infer = torch.zeros_like(self.lut) 
        lut_infer[self.lut > 0] = 1
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
            output = (lut_p * torch.sigmoid(self.lut*LUT_SCALE)).sum(-1)
        return output


class MyCNN(nn.Module):
    def __init__(self, inputs_d, kernal_d, kernal_a, stride):
        super(MyCNN, self).__init__()
        self.stride = stride
        self.kernal_d = kernal_d
        connect_kernal = torch.randn(kernal_d*SIX, inputs_d, kernal_a, kernal_a) *CONNECT_RANDN_K
        self.connect_kernal = torch.nn.Parameter(connect_kernal)
        self.lut_layer = LutLayer(kernal_d)
        self.conv_bn = nn.BatchNorm2d(kernal_d)
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
        x = self.lut_layer(x, infer)


        x_ori = x.clone()
        r = torch.cuda.FloatTensor(x.shape).uniform_()
        x = (x > r).float()
        p =  (x + x_ori - 1).abs()
        p_log = p.log()

        x = x.view(output_shape).permute(0,3,1,2)
        p_log = p_log.view(output_shape).view(output_shape[0], -1).sum(-1)
        return x, p_log


class NetA(nn.Module):
    def __init__(self, input_size=784):
        super(NetA, self).__init__()
        self.cnn1 = MyCNN(inputs_d=1, kernal_d=32, kernal_a=8, stride = 4) #(1,1,1280)1280
        self.cnn2 = MyCNN(inputs_d=32, kernal_d=64, kernal_a=3, stride = 1) #(1,1,1280)1280
        self.cnn3 = MyCNN(inputs_d=64, kernal_d=1000, kernal_a=4, stride = 1) #(1,1,1280)1280

    def forward(self, inputs, infer=False):
        quant_state = True
        x0 = inputs
        x1,p_log1 = self.cnn1(x0,infer,quant=quant_state)
        x2,p_log2 = self.cnn2(x1,infer,quant=quant_state)
        x3,p_log3 = self.cnn3(x2,infer,quant=quant_state)
        return (x0, x1,x2,x3), (p_log1, p_log2, p_log3)

class NetC(nn.Module):
    def __init__(self):
        super(NetC, self).__init__()
        self.C0 = torch.nn.Sequential( 
                        torch.nn.Conv2d(1, 32, 8, stride=4),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace = True),
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

    def forward(self, x0, x1, x2, x3, labels):
        s0 = self.C0(x0).squeeze(-1).squeeze(-1)
        s1 = self.C1(x1).squeeze(-1).squeeze(-1)
        s2 = self.C2(x2).squeeze(-1).squeeze(-1)
        s3 = (x3.view(x3.shape[0],-1).view(x3.shape[0], CLASS, -1).mean(-1) - 0.5) * K_SCORE
        return s0,s1,s2,s3

def KL_loss(predict_dist, target_dist):
    predict_dist = predict_dist.exp()
    predict_dist = predict_dist / predict_dist.sum(-1).unsqueeze(-1)
    target_dist = target_dist.exp()
    target_dist = target_dist / target_dist.sum(-1).unsqueeze(-1)
    apple = (predict_dist + 1e-6).log() - (target_dist + 1e-6).log()
    loss = -(target_dist * apple).sum(-1)
    return loss

def get_loss(s0, s1, s2, s3, labels):
    loss_function = nn.CrossEntropyLoss(reduction='none')
    l0 = loss_function(s0, labels)
    l1 = loss_function(s1, labels)
    l2 = loss_function(s2, labels)
    l3 = loss_function(s3, labels)
    return l0, l1, l2, l3

def get_loss_acc(x_list, labels, p_log_list):
    global DEBUG
    p_log1, p_log2, p_log3 = p_log_list
    x0, x1, x2, x3 = x_list
    s0, s1, s2, s3 = netC(x0, x1, x2, x3, labels)
    l0, l1, l2, l3 = get_loss(s0, s1, s2, s3, labels)
    critic_loss = 0
    critic_loss += KL_loss(s2, s3.detach())
    critic_loss += KL_loss(s1, s2.detach())
    critic_loss += KL_loss(s0, s1.detach())
    critic_loss = critic_loss.mean()
    actor_loss = (l3 - l2) * p_log3 + (l2 - l1) * p_log2 + (l1 - l0) * p_log1
    actor_loss = actor_loss.mean()
    accurate = (s3.argmax(-1) == labels).float().mean() * 100
    return actor_loss, critic_loss,  accurate

def get_fpga_acc(train=True):
    total_correct = 0
    total_items = 0
    if train:
        loader = trainloader
    else:
        loader = testloader
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data[0].cuda(), data[1].cuda()
            x_list, p_log_list = netA(images,infer=True)
            _, _, accurate = get_loss_acc(x_list, labels, p_log_list)
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
        print(' test_acc:%8.3f%%'%(acc))
        if IF_WANDB:
            wandb.log({'test_acc':acc})

netA = NetA().cuda()
netC = NetC().cuda()
optimizerA = optim.Adam(netA.parameters())
optimizerC = optim.Adam(netC.parameters())

def get_L2_loss():
    a = ((netA.cnn1.lut_layer.lut * LUT_SCALE) ** 2).mean()
    b = ((netA.cnn2.lut_layer.lut * LUT_SCALE) ** 2).mean()
    c = ((netA.cnn3.lut_layer.lut * LUT_SCALE) ** 2).mean()
    return a+b+c

DEBUG = False
actor_step_counts = 0
for epoch in range(1000000):
    print(epoch)
    if epoch % 5 == 0:
        optimizerC = optim.Adam(netC.parameters())
    if epoch % 5 == 2:
        optimizerA = optim.Adam(netA.parameters())
    for i, data in enumerate(trainloader, 0):
        images,labels = data[0].cuda(), data[1].cuda()
        x_list, p_log_list = netA(images)
        actor_loss, critic_loss, acc = get_loss_acc(x_list, labels, p_log_list)
        if epoch % 5 > 1:
            loss = actor_loss# + get_L2_loss() * 0.01
            loss.backward()
        else:
            loss = critic_loss
            optimizerC.zero_grad()
            loss.backward()
            optimizerC.step()
        if i % 50 == 0:
            optimizerA.step()
            optimizerA.zero_grad()

            DEBUG = 1
            print('%5d     acc:%8.3f    actor:%8.4f     critic:%8.4f   L2:%8.4f'\
                    %(i,acc,actor_loss, critic_loss, get_L2_loss()))
            if IF_WANDB:
                wandb.log({'acc':acc})
    get_fpga_acc(train = True)
    get_fpga_acc(train = False)
    if epoch % 50 == 0 and IF_SAVE:
        torch.save(netA.state_dict(), './models/ac_%d.model'%epoch+100000)
        #torch.save(netC.state_dict(), './models/mnistC%d.model'%epoch)


