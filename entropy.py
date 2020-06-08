import mnist_web
import select
import numpy as np
import random
import sys
from dataloader import DataLoader
from cradle import Cradle
import torch

IF_WANDB = 0
if IF_WANDB:
    import wandb
    wandb.init()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class Training(): 
    def __init__(self, inputs_n ,cradle_n=50, repro_n = 500, repro_bunch = 20, cuda = False):
        self.cradle = Cradle(cradle_n, inputs_n, mutation_rate = 0.005,
                fading_rate = 0.99995,cuda=cuda)
        self.dl = DataLoader(train = True, cuda=cuda)
        _, self.prior_labels = self.dl.get_all()
        self.prior_n = 0
        self.repro_n = repro_n
        self.repro_bunch = repro_bunch
        self.best_result = {'loss':9999}
        self.cuda = cuda

    def analyse_bins(self,bins):#analyse:[1]correct_rate [2]global_entroy [3]single_w_entroy
        #classificaion by outputs
        class1 = bins.reshape((-1,10)).float() + 0.0001
        class1_sum = torch.sum(class1,1).reshape(-1,1)
        global_entorypy = -torch.sum(class1 * torch.log(class1 / class1_sum))
        global_entorypy /= torch.sum(class1)
        class1_max, _ = torch.max(class1,1)
        correct_rate = torch.sum(class1_max) / torch.sum(class1_sum)
        bins_min = torch.min(class1_sum) 
        bins_max = torch.max(class1_sum)
        loss1 = bins_max / bins_min
        return global_entorypy, correct_rate, bins.reshape(-1,10)
        #print(correct_rate,global_entorypy)

    def train_one_bunch(self):
        inputs, _ = self.dl.get_all()
        bunch_w = self.cradle.get_w(self.repro_bunch)
        bunch_w = bunch_w.permute(1, 0)
        outputs = torch.mm(inputs, bunch_w)
        outputs = outputs.type(torch.int32)
        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0
        new_labels = self.prior_labels + outputs * 10 * (2 ** self.prior_n)
        bunch_loss = torch.zeros((self.repro_bunch,1))
        if self.cuda:
            bunch_loss = bunch_loss.cuda()
        for i in range(self.repro_bunch):
            label = new_labels[:,i]
            bins = torch.bincount(label,minlength=10 * (2 ** self.prior_n) * 2)
            bunch_loss[i],correct_rate,class1 = self.analyse_bins(bins)
            if bunch_loss[i] < self.best_result['loss']:
                self.best_result['loss'] = bunch_loss[i]
                self.best_result['w'] = bunch_w[i]
                self.best_result['label'] = new_labels[:,i].reshape(-1,1)
                self.best_result['correct_rate'] = correct_rate
                self.best_result['bins'] = class1
        bunch_w = bunch_w.permute(1, 0)
        self.cradle.pk(bunch_w,bunch_loss)

    def accumulate(self):
        self.prior_labels = self.best_result['label'] 
        self.prior_n += 1
        self.cradle.from_strach()
        self.best_result = {'loss':9999}

    def show_loss(self, i = 0 ,show_type = 0):
        if show_type == 0:
            print('%3d %6.3f   %6.3f%%'%(i,self.best_result['loss'],self.best_result['correct_rate'] * 100))
        if show_type == 1:
            print('%6.3f   %6.3f%%'%(self.best_result['loss'],self.best_result['correct_rate'] * 100))
            print(self.best_result['bins'])
    
    def adjust_fading_rate(self,j):
        if j == 0:
            self.cradle.set_fading_rate(0.99995)
        if j == 10:
            self.cradle.set_fading_rate(0.99999)
        if j == 30:
            self.cradle.set_fading_rate(0.999995)
        if j == 50:
            self.cradle.set_fading_rate(0.999999)
        if j == 100:
            self.cradle.set_fading_rate(0.9999995)

CRADLE_N = 50
INPUTS_N = 784 
REPRO_N = 500
REPRO_BUNCH = 50

t = Training(inputs_n = INPUTS_N ,cradle_n= CRADLE_N,\
        repro_n = CRADLE_N, repro_bunch = REPRO_BUNCH,cuda=True)
for i in range(10):
    for j in range(100):
        t.adjust_fading_rate(j)
        for k in range(REPRO_N//REPRO_BUNCH):
            t.train_one_bunch()
        t.show_loss(show_type=0, i=j)
    t.show_loss(show_type=1)
    t.accumulate()
