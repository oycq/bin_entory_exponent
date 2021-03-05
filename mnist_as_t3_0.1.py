import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
from mnist_dataset import trainloader, testloader
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
CONNECT_RANDN_K = 0.01
LUT_SCALE = 50
VISAUL = 0

if IF_WANDB:
    import wandb
    wandb.init(project = 'reinforce')#, name = '.')



#for i, data in enumerate(trainloader, 0):
#    images, labels = data
#    x = images[0].float()
#    images = x.permute(1,2,0)
#    img = torch.zeros((images.shape[0],images.shape[1],3))
#    for i in range(3):
#        for j in range(8):
#            img[:,:,i] += images[:,:,i*8+(7-j)] * (2**j)
#    img = img / 255
#    img = img.cpu().numpy()
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#    cv2.imshow('img0', cv2.resize(img,(640,640),interpolation = cv2.INTER_AREA))
#    key = cv2.waitKey(0)
#    if key == ord('q'):
#        break
#sys.exit(0)



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
            x = F.conv2d(inputs.contiguous(), connect_kernal_infer, stride=self.stride)
        else:
            x = F.conv2d(inputs.contiguous(), connect_kernal, stride=self.stride)
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
        self.cnn1 = MyCNN(inputs_d=1, kernal_d=32, kernal_a=8, stride = 4) #(1,1,1280)1280
        self.cnn2 = MyCNN(inputs_d=32, kernal_d=64, kernal_a=3, stride = 1) #(1,1,1280)1280
        self.cnn3 = MyCNN(inputs_d=64, kernal_d=1000, kernal_a=4, stride = 1) #(1,1,1280)1280
        score_K = torch.zeros(1) + 20
        self.score_K = score_K.cuda()
        #self.score_K = torch.nn.Parameter(score_K)

    def forward(self, inputs, infer=False):
        quant_state = True
        x = inputs
        x = self.cnn1(x,infer,quant=quant_state)
        x = self.cnn2(x,infer,quant=quant_state)
        x = self.cnn3(x,infer,quant=quant_state)
        x = x.view(x.shape[0], -1)
        x = (x - 0.5) * self.score_K
        return x

loss_function = nn.CrossEntropyLoss()
def get_loss_acc(x, labels):
    x = x.view(x.shape[0], CLASS, -1)
    x = x.mean(-1)
    accurate = (x.argmax(-1) == labels).float().mean() * 100
    loss = loss_function(x,labels)
    return loss, accurate

def get_fpga_acc(train=True):
    total_correct = 0
    total_items = 0
    total_loss = 0
    if train:
        loader = trainloader
    else:
        loader = testloader
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            images, labels = data[0].cuda(), data[1].cuda()
            x = net(images,infer=True)
            loss, accurate = get_loss_acc(x, labels)
            total_correct += accurate.item() * labels.shape[0]
            total_loss += loss * labels.shape[0]
            total_items += labels.shape[0]
            if total_items >= TESTING_LEN:
                break
        acc = total_correct / total_items
        loss = total_loss / total_items
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
    kernal_img = kernal_2_img(kernal, (720/2,1440/2))
    for i in range(8):
        kernal_img[1440/8/2*i,:] = 128
    for i in range(3):
        kernal_img[:,720/3/2*i] = 128
    cv2.imshow('img', kernal_img)
    cv2.imwrite('./imgs/%d.jpg'%(idx), kernal_img)

    idx = i + 200000000
    kernal = net.cnn2.connect_kernal / CONNECT_RANDN_K + net.cnn2.appointed_connect
    kernal_img = kernal_2_img(kernal, (720*4/2,1440l/2), 5)
    for i in range(16):
        kernal_img[1440/16/2*i,:] = 128 
    for i in range(8):
        kernal_img[:,720*4/8/2*i] = 128 
    cv2.imshow('img2', kernal_img)


net = Net().cuda()
optimizer = optim.Adam(net.parameters())
#net.load_state_dict(torch.load('./ckj.model'))

def get_cnn_dist(cnn):
    connect_kernal= (cnn.connect_kernal/CONNECT_RANDN_K+cnn.appointed_connect).exp()
    connect_kernal_shape = connect_kernal.shape
    connect_kernal = connect_kernal.view(cnn.kernal_d*SIX, -1)
    connect_kernal = connect_kernal / connect_kernal.sum(-1).unsqueeze(-1)
    connect_kernal = connect_kernal.view(connect_kernal_shape)

    connect_kernal = connect_kernal.permute(1,0,2,3)
    connect_kernal = connect_kernal.reshape(connect_kernal.shape[0], -1)
    dist = connect_kernal.sum(-1) 
    dist = dist / dist.sum(-1).unsqueeze(-1)
    dist,_ = dist.sort(descending=True)
   # dist_entropy = (dist.log() * dist).sum()
   # max_entropy = torch.zeros_like(dist) + 1.0 / dist.shape[0]
   # max_entropy = (max_entropy.log() * max_entropy).sum()
   # delta_entroy = dist_entropy - max_entropy
    return dist

def get_dist():
    dist1 = get_cnn_dist(net.cnn2) 
    print(dist1)


def debug_max():
    print('1: %8.3f,%8.3f'%(net.cnn1.connect_kernal.abs().max(),net.cnn1.lut_layer.lut.abs().max()))
    print('2: %8.3f,%8.3f'%(net.cnn2.connect_kernal.abs().max(),net.cnn2.lut_layer.lut.abs().max()))


for epoch in range(1000000):
    print(epoch)
    for i, data in enumerate(trainloader, 0):
        images,labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        x = net(images)
        loss,acc = get_loss_acc(x,labels)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('%5d  %7.3f %7.4f  %7.4f'%(i,acc,loss, net.score_K))
            if IF_WANDB:
                wandb.log({'acc':acc})
            if VISAUL:
                visual(i)
    get_fpga_acc(train = True)
    get_fpga_acc(train = False)
    debug_max()
    get_dist()
    if epoch % 3 == 0 and IF_SAVE:
        torch.save(net.state_dict(), './models/cifar_%d.model'%epoch)
        #death(i)


