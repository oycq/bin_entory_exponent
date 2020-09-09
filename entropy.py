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
save_npy_name = 'similar_more_train.npy'
if IF_WANDB:
    import wandb
    wandb.init()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CRADLE_SIZE = 50
INPUT_SIZE = 784
REPRO_SIZE = 1 
CUDA = 1
top_k_rate = 0.5

dl_test = DataLoader(False,CUDA)
images_t,labels_t = dl_test.get_all()
dl = DataLoader(True,CUDA)
images,labels = dl.get_all()
images, labels = images[:18000],labels[:18000]
cradle = Cradle(CRADLE_SIZE, INPUT_SIZE, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)


accumulate = torch.zeros((labels.shape[0],labels.shape[0]),dtype = torch.float32)
accumulate -= 100 * torch.eye(labels.shape[0])
accumulate_t = torch.zeros((labels_t.shape[0],labels.shape[0]),dtype = torch.float32)
to_save = np.zeros((100,784),dtype = np.float32)

if CUDA:
    accumulate = accumulate.cuda()
    accumulate_t = accumulate_t.cuda()

def get_images_output(brunch_w, images):
    if len(brunch_w.shape) == 1:
        w = brunch_w.unsqueeze(1)
    else:
        w =  brunch_w.t()#[784*N]
    o = images.mm(w)#[60000*N]
    top_k_elements,_ = torch.topk(o.flatten(), int(o.numel()*top_k_rate))
    throat = top_k_elements.min()
    select = (o >= throat)
    o[select] = 1
    select = ~select
    o[select] = 0
    return o


def get_similarity_table(o1,o2):
    r = o1.t().unsqueeze(2).bmm(o2.t().unsqueeze(1))
    if not CUDA:
        r = r.float()
    return r


def get_classfication_score_table(similar_table, labels, accumulate):
    r = similar_table + accumulate
    r = 2 ** r #[N,10000,10000]
    l = labels.repeat(r.shape[0],1,1)
    if not CUDA:
        l = l.float()
    r = r.bmm(l) #[N,60000,10]
    r /= labels.sum(0).unsqueeze(0).unsqueeze(0)
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
    for i in range(5000):
        brunch_w = cradle.get_w(REPRO_SIZE)
        o = get_images_output(brunch_w, images)
        r = get_similarity_table(o,o)
        r = get_classfication_score_table(r, labels, accumulate)
        r = get_loss(r, labels)
        cradle.pk(brunch_w,r)
        if i % 500 == 0:
            print('loss:%8.4f'%cradle.get_best()[0].item())
            show_gather(o[:,0].unsqueeze(1),labels)

    w = cradle.get_best()[1]
    if IF_SAVE:
        to_save[j,:] = w.cpu().numpy()
        np.save(save_npy_name,to_save)
    o = get_images_output(w, images)
    similar_table = get_similarity_table(o, o)
    show_gather(o, labels)
    r = get_classfication_score_table(similar_table, labels, accumulate)
    show_accuarcate(r, labels)
    accumulate += similar_table[0]
    del r,similar_table

    o_t = get_images_output(w, images_t)
    similar_table_t = get_similarity_table(o_t,o)
    r = get_classfication_score_table(similar_table_t, labels, accumulate_t)
    show_accuarcate(r, labels_t,train=False)
    accumulate_t += similar_table_t[0]
    del r,similar_table_t

    #r= get_similarity_table(w, images_test)
    #r = get_classfication_score_table(r, labels_test, accumulate_test)
    #show_accuarcate(r, labels_test)
    #accumulate_test += get_similarity_table(w, images_test)[0]
