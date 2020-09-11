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
IF_SAVE = 1
save_npy_name = 'exp_similar_fast.npy'
if IF_WANDB:
    import wandb
    wandb.init()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CRADLE_SIZE = 50
INPUT_SIZE = 784
REPRO_SIZE = 3 
CUDA = 1
top_k_rate = 0.5

dl = DataLoader(True,CUDA)
images,labels = dl.get_all()
dl_test = DataLoader(False,CUDA)
images_t,labels_t = dl_test.get_all()
cradle = Cradle(CRADLE_SIZE, INPUT_SIZE, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)


accumulate = torch.zeros((labels.shape[0],0),dtype = torch.float32)
accumulate_t = torch.zeros((labels_t.shape[0],0),dtype = torch.float32)
to_save = np.zeros((100,784),dtype = np.float32)

if CUDA:
    accumulate = accumulate.cuda()
    accumulate_t = accumulate_t.cuda()

def get_images_output(brunch_w, images, train=True):
    if len(brunch_w.shape) == 1:
        w = brunch_w.unsqueeze(1)
    else:
        w =  brunch_w.t()#[784*N]
    o = images.mm(w)#[60000*N]
    o[o>0] = 1
    o[o<0] = 0
    #top_k_elements,_ = torch.topk(o.flatten(), int(o.numel()*top_k_rate))
    #throat = top_k_elements.min()
    #select = (o >= throat)
    #o[select] = 1
    #select = ~select
    #o[select] = 0
    o = o.t().unsqueeze(2)
    if train:
        accum = accumulate
    else:
        accum = accumulate_t
    accum = accum.repeat(brunch_w.shape[0],1,1)
    o = torch.cat((accum,o), 2)
    return o


def get_classfication_score_table(o1, o2, labels):
    #assume o1->[N,20000,4]
    #                (                   c                 )
    #                (      a      )           (      b    )       
    #    o1          o2t          o2           o2t         l
    #[N,20000,4] [N,4,10000] [N,10000,4]  [N,4,10000]  [N,10000,10]
    o1t = o1.transpose(1,2)
    o2t = o2.transpose(1,2)
    l = labels.repeat(o1.shape[0],1,1)
    if not CUDA:
        l = l.float()
    a = o2t.bmm(o2)
    b = o2t.bmm(l)
    c = a.bmm(b)
    r1 = o1.bmm(b)
    r2 = o1.bmm(c) / o2.shape[1]
    r = r1
    r += 1
    r /= labels.sum(0).unsqueeze(0).unsqueeze(0)
    r = torch.exp(r)
    return r

def show_gather(images_o, labels):
    r = images_o.t().mm(labels)
    output_str = '                '
    for i in range(10):
       output_str += '%7d'%(r[0][i])
    print(output_str)


def get_loss(class_s_table, labels):
    r = class_s_table / class_s_table.sum(2).unsqueeze(2)
    r = -torch.sum(torch.log(r) * labels,2)
    r = torch.mean(r, 1)
    return r

def show_accuarcate(r, labels, train=True):#r:classfication_score_table
    a = torch.argmax(r[0],1)
    b = labels.argmax(1)
    accuarcate = torch.mean((a==b).float())*100
    if train:
        print('Train accuarcate:%6.2f%%'%(accuarcate))
        print(r[0,:3,:10])
        print(labels[:3])
        if IF_WANDB:
            wandb.log({'train':accuarcate})
    if not train:
        print('Test accuarcate:%6.2f%%\n\n'%(accuarcate))
        if IF_WANDB:
            wandb.log({'test':accuarcate})


def img_show(img):#[784]
    for j in range(28):
        a = img[j*28:j*28+28]
        st = ''
        for k in range(28):
            if a[k] == -1:
                st+='**'
            else:
                st+='  '
        print(st)



for j in range(100):
    print(j)
    cradle.from_strach()
    for i in range(2500):
        brunch_w = cradle.get_w(REPRO_SIZE)
        o = get_images_output(brunch_w, images)
        r = get_classfication_score_table(o, o, labels)
        r = get_loss(r, labels)
        cradle.pk(brunch_w,r)
        if i % 500 == 0:
            print('loss:%8.4f'%cradle.get_best()[0].item())
            show_gather(o[0,:,-1].unsqueeze(1),labels)
    w = cradle.get_best()[1]
    if IF_SAVE:
        to_save[j,:] = w.cpu().numpy()
        np.save(save_npy_name,to_save)
    w = w.unsqueeze(0)
    o = get_images_output(w, images)
    show_gather(o[0,:,-1].unsqueeze(1),labels)
    r = get_classfication_score_table(o, o, labels)
    show_accuarcate(r, labels)
    accumulate = torch.cat((accumulate, o[0:,:,-1].reshape(-1,1)),1)

    ot = get_images_output(w, images_t, train=False)
    r = get_classfication_score_table(ot, o, labels)
    show_accuarcate(r, labels_t,train=False)
    accumulate_t = torch.cat((accumulate_t, ot[0:,:,-1].reshape(-1,1)),1)
