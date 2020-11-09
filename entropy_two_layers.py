import mnist_web
import select
import numpy as np
import random
import sys
from dataloader import DataLoader
from cradle import Cradle
import torch
import time

IF_WANDB = 0
IF_SAVE = 0
if IF_WANDB:
    import wandb
    wandb.init()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CRADLE_SIZE = 50
INPUT_SIZE = 50
REPRO_SIZE = 5 
CUDA = 1

dl = DataLoader(False,CUDA)
images,labels = dl.get_all()
w = np.load('./exp1_4.npy')[:INPUT_SIZE]
w = torch.from_numpy(w).type(torch.float32).cuda().t()[:,:INPUT_SIZE]
images = images.mm(w)
images[images>0] = 1
images[images<0] = 0
cradle = Cradle(CRADLE_SIZE, INPUT_SIZE, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)

accumulate = torch.zeros((labels.shape[0],labels.shape[0]),dtype = torch.float32) + 0.01
to_save = np.zeros((60,INPUT_SIZE),dtype = np.float32)

if CUDA:
    accumulate = accumulate.cuda()

for j in range(50):
    print(j)
    cradle.from_strach()
    for i in range(1500):
        brunch_w =  cradle.get_w(REPRO_SIZE)
        w =  brunch_w.t()#[784*N]
        o = images.mm(w)#[60000*N]
        o[o>0] = 1
        o[o<=0] = -1
        o = o.t()#[N*60000]
        r = o.unsqueeze(2).bmm(o.unsqueeze(1))
        r /= 2
        r += 0.5
        if not CUDA:
            r = r.float()
        r += accumulate
        #r *= r #[N,10000,10000]
        r = 1.4 ** r #[N,10000,10000]
        l = labels.repeat(REPRO_SIZE,1,1)
        if not CUDA:
            l = l.float()
        r = r.bmm(l) #[N,60000,10]
        if i == 1400:
            a = torch.argmax(r[0],1)
            b = labels.argmax(1)
            print('accuarcate:%6.2f%%'%(torch.mean((a==b).float())*100))
            print(r[0,:3,:10])
            print(labels[:3])
        r = r / r.sum(2).unsqueeze(2)
        r = -torch.sum(torch.log(r) * labels,2)
        r = torch.mean(r, 1)
        cradle.pk(brunch_w,r)
        if i % 100 == 0:
            print(cradle.get_best()[0])

    w = cradle.get_best()[1]
    if IF_SAVE:
        to_save[j,:] = w.cpu().numpy()
        np.save('exp1_4.npy',to_save)
    w = w.unsqueeze(1)
    o = images.mm(w)#[60000*N]
    o[o>0] = 1
    o[o<=0] = -1
    r = o * o.t()
    r[r==-1] = 0
    accumulate += r
    del r
