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
if IF_WANDB:
    import wandb
    wandb.init()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CRADLE_SIZE = 50
INPUT_SIZE = 784
REPRO_SIZE = 20
CUDA = 1

dl = DataLoader(False,CUDA)
images,labels = dl.get_all()
cradle = Cradle(CRADLE_SIZE, INPUT_SIZE, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)

accumulate = torch.zeros((REPRO_SIZE,labels.shape[0],10),dtype = torch.float32) + 1
if CUDA:
    accumulate = accumulate.cuda()

for j in range(20):
    print(j)
    cradle.from_strach()
    for i in range(5000):
        time1 = time.time() * 1000
        brunch_w =  cradle.get_w(REPRO_SIZE)
        time2 = time.time() * 1000
        w =  brunch_w.t()#[784*N]
        o = images.mm(w)#[60000*N]
        o[o>0] = 1
        o[o<=0] = -1

        o = o.t()#[N*60000]
        r = o.mm(labels).unsqueeze(1)#[N*1*10]
        result = o.unsqueeze(2).bmm(r)#[N*60000*10]
        if CUDA:
            result = (result + labels.sum(0)) / 2
        else:
            result = (result + labels.sum(0).int()).float() / 2
        result += accumulate
        torch.cuda.synchronize()

        time3 = time.time() * 1000
        #caculate entroy
        result = result / result.sum(2).unsqueeze(2)
        #result = -torch.sum(result * torch.log(result),2)
        result = -torch.sum(torch.log(result) * labels,2)
        result = torch.mean(result,1)
        torch.cuda.synchronize()

        time4 = time.time() * 1000
        cradle.pk(brunch_w,result)
        time5 = time.time() * 1000

        #print('%8.4f  %8.4f  %8.4f  %8.4f'%(time2-time1,time3-time2,time4-time3,time5-time4))
        if i % 100 == 0:
            print(cradle.get_best()[0])

    brunch_w =  cradle.get_w(REPRO_SIZE)
    for i in range(REPRO_SIZE):
        brunch_w[i] = cradle.get_best()[1]
    w =  brunch_w.t()#[784*N]
    o = images.mm(w)#[60000*N]
    o[o>0] = 1
    o[o<=0] = -1
    o = o.t()#[N*60000]
    r = o.mm(labels).unsqueeze(1)#[N*1*10]
    result = o.unsqueeze(2).bmm(r)#[N*60000*10]
    if CUDA:
        result = (result + labels.sum(0)) / 2
    else:
        result = (result + labels.sum(0).int()).float() / 2
    accumulate += result
    print(accumulate)


sys.exit(0)

INPUTS_N = 784 
REPRO_N = 5000
REPRO_BUNCH = 50
J = 10
CUDA = 1
LEAVES_N = 256
SAVE_PATH = './maxsume_leaf.npy'

t = Training(inputs_n = INPUTS_N ,cradle_n= CRADLE_N,\
        repro_n = CRADLE_N, repro_bunch = REPRO_BUNCH,cuda=CUDA)
need_save = np.zeros((INPUTS_N, LEAVES_N),dtype = int)

for leaves_n in range(LEAVES_N):
    print(leaves_n)
    correct_account = 0
    for j in range(J):
        t.adjust_fading_rate(j)
        for k in range(REPRO_N//REPRO_BUNCH):
            t.train_one_bunch()
        t.show_loss(show_type=0, i=j)
    t.dl.bifurcate(t.best_result['outputs'],t.best_result['lr_entropy'],\
            t.best_result['bins'])
    t.validation()
    need_save[:,leaves_n] = t.best_result['w'].cpu()
    np.save(SAVE_PATH, need_save)
    t.reset()
    t.dl.print_statue()
