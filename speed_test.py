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

    def analyse_bins(self,ori_bins):#analyse:[1]correct_rate [2]global_entroy [3]single_w_entroy
        #classificaion by outputs
        bins = ori_bins.reshape((-1,10)).float() + 0.000001
        bins_sum = torch.sum(bins,1).reshape(-1,1)
        global_entorypy = -torch.sum(bins* torch.log(bins/ bins_sum))
        global_entorypy /= torch.sum(bins)

        bins2 = torch.sum(bins.reshape(2,-1,10),1)
        bins2_sum = torch.sum(bins2,1).reshape(-1,1)
        bins2_entorypy = -torch.sum(bins2* torch.log(bins2/ bins2_sum))
        bins2_entorypy /= torch.sum(bins2)

        bins_max_ax1, _ = torch.max(bins,1)
        correct_rate = torch.sum(bins_max_ax1) / torch.sum(bins_sum)
        bins_min = torch.min(bins_sum) 
        bins_max = torch.max(bins_sum)
        loss_k = bins_max / bins_min
        loss = bins2_entorypy + loss_k * 0.1
        return loss, correct_rate, ori_bins.reshape(-1,10),\
                bins2,bins2_entorypy,global_entorypy,loss_k
        #print(correct_rate,global_entorypy)

    last_time = time.time() * 1000 
    def pt(self,name):
        torch.cuda.synchronize()
        t = time.time() * 1000
        print('%10s  %10.3f'%(name,t-self.last_time))
        self.last_time = t


    def train_one_bunch(self):
        self.pt('init')
        inputs, _ = self.dl.get_all()
        bunch_w = self.cradle.get_w(self.repro_bunch)
        bunch_w = bunch_w.permute(1, 0)
        outputs = torch.mm(inputs, bunch_w)
        outputs = outputs.type(torch.int32)
        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0
        self.pt('1')
        new_labels = self.prior_labels + outputs * 10 * (2 ** self.prior_n)
        bunch_loss = torch.zeros((self.repro_bunch,1))
        self.pt('2')
        if self.cuda:
            bunch_loss = bunch_loss.cuda()
        self.pt('---')
        for i in range(self.repro_bunch):
            if i % 100 == 0:
                self.pt('....')
            label = new_labels[:,i]
            bins = torch.bincount(label,minlength=10 * (2 ** self.prior_n) * 2)
            if i % 100 == 0:
                self.pt('++++')
            bunch_loss[i],correct_rate,bins,bins2,bins2_entorypy,\
                    global_entorypy,loss_k = self.analyse_bins(bins)
            if i % 100 == 0:
                self.pt('----')
            if bunch_loss[i] < self.best_result['loss']:
                self.best_result['loss'] = bunch_loss[i]
                self.best_result['w'] = bunch_w[i]
                self.best_result['label'] = new_labels[:,i].reshape(-1,1)
                self.best_result['correct_rate'] = correct_rate
                self.best_result['bins'] = bins 
                self.best_result['bins2'] = bins2
                self.best_result['global_entorypy'] = global_entorypy 
                self.best_result['bins2_entorypy'] = bins2_entorypy 
                self.best_result['loss_k'] = loss_k
                self.pt('---%d'%i)
            if i % 100 == 0:
                self.pt('----')
        self.pt('3')
        bunch_w = bunch_w.permute(1, 0)
        self.cradle.pk(bunch_w,bunch_loss)
        self.pt('4')

    def accumulate(self):
        self.prior_labels = self.best_result['label'] 
        self.prior_n += 1
        self.cradle.from_strach()
        self.best_result = {'loss':9999}

    def show_loss(self, i = 0 ,show_type = 0):
        if show_type == 0:
            print('%3d %6.3f   %6.3f%%'%(i,self.best_result['loss'],\
                    self.best_result['correct_rate'] * 100))
        if show_type == 1:
            print('%6.3f   %6.3f%%'%(self.best_result['loss'],self.best_result['correct_rate'] * 100))

            print(self.best_result['bins'])
            print(torch.sum(self.best_result['bins'],1))
            print(self.best_result['bins'].shape[0])

            print(self.best_result['bins2'])
            print('bins2_entorypy',self.best_result['bins2_entorypy'])
            print('loss_k',self.best_result['loss_k'])
    
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
REPRO_N = 5000
REPRO_BUNCH = 500

t = Training(inputs_n = INPUTS_N ,cradle_n= CRADLE_N,\
        repro_n = CRADLE_N, repro_bunch = REPRO_BUNCH,cuda=True)
for i in range(10):
    for j in range(3):
        t.adjust_fading_rate(j)
        for k in range(REPRO_N//REPRO_BUNCH):
            t.train_one_bunch()
        t.show_loss(show_type=0, i=j)
    t.show_loss(show_type=1)
    t.accumulate()
