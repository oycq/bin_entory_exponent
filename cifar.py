import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
import cifar_dataset as my_dataset 
import cv2
import numpy as np
import random

torch.manual_seed(0)
random.seed()

IF_WANDB = 0
IF_SAVE = 0
SIX = 6
BATCH_SIZE = 100
WORKERS = 15
CLASS = 10
TESTING_LEN = 10000
LUT_RANDN_K = 0
CONNECT_RANDN_K = 0.1
LUT_SCALE = 50
VISAUL = 0

if IF_WANDB:
    import wandb
    wandb.init(project = 'lut_cifar')#, name = '.')



dataset = my_dataset.MyDataset(train = True, margin = 2, noise_rate = 0.01)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()

#images, labels = data_feeder.feed()
#x = images[0].float()
#images = x.permute(1,2,0)[:,:,:8]
#img = 0
#for i in range(8):
#    img += images[:,:,7-i] * (2**i)
#img = img / 256
#img = img.cpu().numpy()
#cv2.imshow('img0', cv2.resize(img,(640,640),interpolation = cv2.INTER_AREA))
#cv2.waitKey(0)


class Quantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        r = torch.cuda.FloatTensor(input.shape).uniform_()
        return (input >= r).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


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
        self.quantized = Quantized.apply
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
        connect_kernal_shape = self.connect_kernal.shape
        connect_kernal= (self.connect_kernal/CONNECT_RANDN_K+self.appointed_connect).exp()
        connect_kernal = connect_kernal.view(self.kernal_d*SIX, -1)
        connect_kernal = connect_kernal / connect_kernal.sum(-1).unsqueeze(-1)
        connect_kernal = connect_kernal.view(connect_kernal_shape)
        connect_kernal_infer = self.get_infer_kernal()
        if infer:
            x = F.conv2d(inputs, connect_kernal_infer, stride=self.stride)
        else:
            x = F.conv2d(inputs, connect_kernal, stride=self.stride)
        x = x.permute(0,2,3,1)
        output_shape = (x.shape[0],x.shape[1],x.shape[2],self.kernal_d)
        x = x.reshape(-1, self.kernal_d, SIX)
        x = self.lut_layer(x, infer)
        if quant:
            x = self.quantized(x)
        x = x.view(output_shape).permute(0,3,1,2)
        return x


class Net(nn.Module):
    def __init__(self, input_size=784):
        super(Net, self).__init__()
        self.cnn1 = MyCNN(inputs_d=24, kernal_d=8,  kernal_a=1, stride = 1) #(32,32,8)8192
        self.cnn2 = MyCNN(inputs_d=8, kernal_d=16,  kernal_a=6, stride = 2) #(14,14,16)3136
        self.cnn3 = MyCNN(inputs_d=16, kernal_d=32,  kernal_a=6, stride = 1) #(9,9,32)2592
        self.cnn4 = MyCNN(inputs_d=32, kernal_d=64, kernal_a=4, stride = 1) #(6,6,64)2304
        self.cnn5 = MyCNN(inputs_d=64, kernal_d=1280, kernal_a=6, stride = 1) #(1,1,1280)1280
        self.cnn6 = MyCNN(inputs_d=1280, kernal_d=1280, kernal_a=1, stride = 1) #(1,1,1280)1280
        self.cnn7 = MyCNN(inputs_d=1280, kernal_d=1280, kernal_a=1, stride = 1) #(1,1,1280)1280
        score_K = torch.zeros(1) + 20
        self.score_K = score_K.cuda()
        #self.score_K = torch.nn.Parameter(score_K)

    def forward(self, inputs, infer=False):
        quant_state = True
        x = inputs
        x = self.cnn1(x,infer,quant=quant_state)
        x = self.cnn2(x,infer,quant=quant_state)
        x = self.cnn3(x,infer,quant=quant_state)
        x = self.cnn4(x,infer,quant=quant_state)
        x = self.cnn5(x,infer,quant=quant_state)
        x = self.cnn6(x,infer,quant=quant_state)
        x = self.cnn7(x,infer,quant=quant_state)
        x = x.view(x.shape[0], -1)
        x = (x - 0.5) * self.score_K
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

def get_fpga_acc(train=True):
    acc = 0
    with torch.no_grad():
        for i in range(TESTING_LEN//BATCH_SIZE):
            if train:
                images, labels = data_feeder.feed()
            else:
                a = i * BATCH_SIZE
                b = i * BATCH_SIZE + BATCH_SIZE
                images, labels = images_t[a:b], labels_t[a:b]
            x = net(images,infer=True)
            loss, accurate = get_loss_acc(x, labels)
            acc += accurate.item() * 1.0 * BATCH_SIZE / TESTING_LEN
    if train:
        print('train_acc:%8.3f%%   train_loss:%8.3f'%(acc,loss))
        if IF_WANDB:
            wandb.log({'train_acc':acc})
    else:
        print(' test_acc:%8.3f%%    test_loss:%8.3f'%(acc,loss))
        if IF_WANDB:
            wandb.log({'test_acc':acc})

def kernal_2_img(kernal, resize,scale=1):
    kernal = kernal.view(kernal.shape[0],-1)
    kernal = kernal.exp()
    kernal = kernal / kernal.sum(-1).unsqueeze(-1)
    kernal = kernal * 255 * scale
    kernal[kernal > 255] = 255
    kernal = kernal.detach().cpu().numpy().astype(np.uint8)
    kernal_img = cv2.resize(kernal,resize,interpolation = cv2.INTER_AREA)
    return kernal_img

def visual(i):
    idx = i + 100000000
    kernal = net.cnn1.connect_kernal / CONNECT_RANDN_K + net.cnn1.appointed_connect
    kernal_img = kernal_2_img(kernal, (720,1440))
    for i in range(8):
        kernal_img[1440/8*i,:] = 128
    for i in range(3):
        kernal_img[:,720/3*i] = 128
    cv2.imshow('img', kernal_img)
    cv2.imwrite('./imgs/%d.jpg'%(idx), kernal_img)

    idx = i + 200000000
    kernal = net.cnn2.connect_kernal / CONNECT_RANDN_K + net.cnn2.appointed_connect
    kernal_img = kernal_2_img(kernal, (720*4,1440), 5)
    for i in range(16):
        kernal_img[1440/16*i,:] = 128 
    for i in range(8):
        kernal_img[:,720*4/8*i] = 128 
    cv2.imshow('img2', kernal_img)

    idx = i + 300000000
    kernal = net.cnn3.connect_kernal / CONNECT_RANDN_K + net.cnn3.appointed_connect
    kernal_img = kernal_2_img(kernal, (720*4,1440), 5)
    for i in range(32):
        kernal_img[1440/32*i,:] = 128 
    for i in range(16):
        kernal_img[:,720*4/16*i] = 128 
    cv2.imshow('img3', kernal_img)
    cv2.waitKey(1)

#    idx = i + 400000000
#    kernal = net.cnn4.connect_kernal / CONNECT_RANDN_K + net.cnn4.appointed_connect
#    kernal_img = kernal_2_img(kernal, (720*4,1440), 1)
#    for i in range(64):
#        kernal_img[1440/64*i,:] = 64
#    for i in range(32):
#        kernal_img[:,720*4/32*i] = 64
#    cv2.imshow('img4', kernal_img)
#
#    idx = i + 400000000
#    kernal = net.cnn5.connect_kernal / CONNECT_RANDN_K + net.cnn5.appointed_connect
#    kernal_img = kernal_2_img(kernal, (720*4,1440), 1)
#    print(kernal.shape)
#    kernal_img = kernal_2_img(kernal, (720*4,1280*6), 5)
##    for i in range(64):
##        kernal_img[1440/64*i,:] = 64
#    for i in range(64):
#        kernal_img[:,720*4/64*i] = 64
#    cv2.imshow('img4', kernal_img)


def get_cnn_loss(cnn):
    connect_kernal= (cnn.connect_kernal/CONNECT_RANDN_K+cnn.appointed_connect).exp()
    connect_kernal_shape = connect_kernal.shape
    connect_kernal = connect_kernal.view(cnn.kernal_d*SIX, -1)
    connect_kernal = connect_kernal / connect_kernal.sum(-1).unsqueeze(-1)
    connect_kernal = connect_kernal.view(connect_kernal_shape)

    connect_kernal = connect_kernal.permute(1,0,2,3)
    connect_kernal = connect_kernal.reshape(connect_kernal.shape[0], -1)
    dist = connect_kernal.sum(-1) + 1
    dist = dist / dist.sum(-1).unsqueeze(-1)
    dist_entropy = (dist.log() * dist).sum()
    max_entropy = torch.zeros_like(dist) + 1.0 / dist.shape[0]
    max_entropy = (max_entropy.log() * max_entropy).sum()
    delta_entroy = dist_entropy - max_entropy
    return delta_entroy

def get_dist_loss():
    loss1 = get_cnn_loss(net.cnn1) * 0.1
    loss2 = get_cnn_loss(net.cnn2)
    loss3 = get_cnn_loss(net.cnn3)
    loss4 = get_cnn_loss(net.cnn4)
    loss5 = get_cnn_loss(net.cnn5)
    loss6 = get_cnn_loss(net.cnn6)
    loss7 = get_cnn_loss(net.cnn7)
    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
    return loss


net = Net().cuda()
optimizer = optim.Adam(net.parameters())
#net.load_state_dict(torch.load('./ckj.model'))

def death(i):
    #k = 1 - 2/(i+10)
    k = 0.999
    net.cnn1.connect_kernal.data *= k
    net.cnn2.connect_kernal.data *= k
    net.cnn3.connect_kernal.data *= k
    net.cnn4.connect_kernal.data *= k
    net.cnn5.connect_kernal.data *= k
    net.cnn6.connect_kernal.data *= k
    net.cnn7.connect_kernal.data *= k
#    net.cnn1.lut_layer.lut.data *= k
#    net.cnn2.lut_layer.lut.data *= k
#    net.cnn3.lut_layer.lut.data *= k
#    net.cnn4.lut_layer.lut.data *= k
#    net.cnn5.lut_layer.lut.data *= k

def debug_max():
    print('1: %8.3f,%8.3f'%(net.cnn1.connect_kernal.abs().max(),net.cnn1.lut_layer.lut.abs().max()))
    print('2: %8.3f,%8.3f'%(net.cnn2.connect_kernal.abs().max(),net.cnn2.lut_layer.lut.abs().max()))
    print('3: %8.3f,%8.3f'%(net.cnn3.connect_kernal.abs().max(),net.cnn3.lut_layer.lut.abs().max()))
    print('4: %8.3f,%8.3f'%(net.cnn4.connect_kernal.abs().max(),net.cnn4.lut_layer.lut.abs().max()))
    print('5: %8.3f,%8.3f'%(net.cnn5.connect_kernal.abs().max(),net.cnn5.lut_layer.lut.abs().max()))
    print('6: %8.3f,%8.3f'%(net.cnn6.connect_kernal.abs().max(),net.cnn6.lut_layer.lut.abs().max()))
    print('7: %8.3f,%8.3f'%(net.cnn7.connect_kernal.abs().max(),net.cnn7.lut_layer.lut.abs().max()))


for i in range(100000000):
    images, labels = data_feeder.feed()
    optimizer.zero_grad()
    x = net(images)
    loss,acc = get_loss_acc(x,labels)
    dist_loss = get_dist_loss()
    loss = loss + dist_loss
    loss.backward()
    if i % 50 == 0:
        print('%5d  %7.3f %7.4f  %7.4f %7.4f'%(i,acc,loss-dist_loss,dist_loss,net.score_K))
        if IF_WANDB:
            wandb.log({'acc':acc})
        if VISAUL:
            visual(i)
    if i % 500 == 0:
        get_fpga_acc(train = True)
        get_fpga_acc(train = False)
        debug_max()
    if i % 4999 == 0 and IF_SAVE:
        torch.save(net.state_dict(), 'ckj_fast_deep2.model')
    optimizer.step()
    #death(i)


