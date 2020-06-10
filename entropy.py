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

class Bifurcate_loader():
    def __init__(self,train=True,cuda=False):
        self.dl = DataLoader(train,cuda)
        images,labels = self.dl.get_all()
        self.images_leaves = [images]
        self.labels_leaves = [labels]
    
    def bifurcate(self,outputs_list):
        new_images_leaves = []
        new_labels_leaves = []
        for i in range(len(outputs_list)):
            images_leaf = self.images_leaves[i]
            labels_leaf = self.labels_leaves[i]
            outputs = outputs_list[i]
            new_images_leaves.append(images_leaf[outputs == 0])
            new_labels_leaves.append(labels_leaf[outputs == 0])
            new_images_leaves.append(images_leaf[outputs == 1])
            new_labels_leaves.append(labels_leaf[outputs == 1])
        self.images_leaves = new_images_leaves
        self.labels_leaves = new_labels_leaves 
        
    def get_leaves_n(self):
        return len(self.images_leaves)
    
    def get_leaf(self,leaf_i):
        return self.images_leaves[leaf_i],self.labels_leaves[leaf_i]


class Training(): 
    def __init__(self, inputs_n ,cradle_n=50, repro_n = 500, repro_bunch = 20, cuda = False):
        self.cradle = Cradle(cradle_n, inputs_n, mutation_rate = 0.005,
                fading_rate = 0.99995,cuda=cuda)
        self.dl = Bifurcate_loader(train = True, cuda=cuda)
        _, self.prior_labels = self.dl.get_leaf(0)
        self.prior_n = 0
        self.repro_n = repro_n
        self.repro_bunch = repro_bunch
        self.best_result = {'loss':9999}
        self.cuda = cuda

    def analyse_bins(self,ori_bins):#analyse:[1]correct_rate [2]global_entroy [3]single_w_entroy
        #classificaion by outputs
        bins = ori_bins.reshape((-1,10)).float() + 0.000001
        bins_sum = torch.sum(bins,1).reshape(-1,1)
        global_entropy = -torch.sum(bins* torch.log(bins/ bins_sum))
        global_entropy /= torch.sum(bins)
        
        worse_entroypy=torch.max(-torch.sum(bins* torch.log(bins/ bins_sum),1)/torch.sum(bins,1))

        bins2 = torch.sum(bins.reshape(2,-1,10),1)
        bins2_sum = torch.sum(bins2,1).reshape(-1,1)
        bins2_entropy = -torch.sum(bins2* torch.log(bins2/ bins2_sum))
        bins2_entropy /= torch.sum(bins2)

        bins_max_ax1, _ = torch.max(bins,1)
        correct_rate = torch.sum(bins_max_ax1) / torch.sum(bins_sum)
        bins_min = torch.min(bins_sum) 
        bins_max = torch.max(bins_sum)
        loss_k = bins_max / bins_min
        #loss = global_entropy + loss_k * 0.1
        #loss = worse_entroypy
        loss = global_entropy 
        return loss, correct_rate, ori_bins.reshape(-1,10),\
                bins2,bins2_entropy,global_entropy,loss_k
        #print(correct_rate,global_entropy)

    def train_one_bunch(self,leaf_i):
        inputs, labels = self.dl.get_leaf(leaf_i)
        bunch_w = self.cradle.get_w(self.repro_bunch)
        bunch_w = bunch_w.permute(1, 0)
        outputs = torch.mm(inputs, bunch_w)
        outputs = outputs.type(torch.int32)
        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0
        new_labels = labels + outputs * 10 * (2 ** self.prior_n)
        bunch_loss = torch.zeros((self.repro_bunch,1))
        if self.cuda:
            bunch_loss = bunch_loss.cuda()
        for i in range(self.repro_bunch):
            label = new_labels[:,i]
            bins = torch.bincount(label,minlength=10 * (2 ** self.prior_n) * 2)
            bunch_loss[i],correct_rate,bins,bins2,bins2_entropy,\
                    global_entropy,loss_k = self.analyse_bins(bins)
            if bunch_loss[i] < self.best_result['loss']:
                self.best_result['loss'] = bunch_loss[i]
                self.best_result['w'] = bunch_w[i]
                self.best_result['label'] = new_labels[:,i].reshape(-1,1)
                self.best_result['correct_rate'] = correct_rate
                self.best_result['bins'] = bins 
                self.best_result['bins2'] = bins2
                self.best_result['global_entropy'] = global_entropy 
                self.best_result['bins2_entropy'] = bins2_entropy 
                self.best_result['loss_k'] = loss_k
                self.best_result['outputs'] = outputs[:,i]
        bunch_w = bunch_w.permute(1, 0)
        self.cradle.pk(bunch_w,bunch_loss)

    def accumulate(self):
        #self.prior_labels = self.best_result['label'] 
        #self.prior_n += 1
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

            #print(self.best_result['bins2'])
            #print('bins2_entropy',self.best_result['bins2_entropy'])
            #print('loss_k',self.best_result['loss_k'])
    
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
REPRO_BUNCH = 50

t = Training(inputs_n = INPUTS_N ,cradle_n= CRADLE_N,\
        repro_n = CRADLE_N, repro_bunch = REPRO_BUNCH,cuda=True)

for bintree_deep in range(6):
    outputs_list = []
    correct_account = 0
    for i in range(2**bintree_deep):
        for j in range(10):
            t.adjust_fading_rate(j)
            for k in range(REPRO_N//REPRO_BUNCH):
                t.train_one_bunch(leaf_i=i)
            t.show_loss(show_type=0, i=j)
        print('bintree_deep:%3d    leaf_i:%3d'%(bintree_deep,i))
        outputs_list.append(t.best_result['outputs'])
        bar = t.best_result['bins']
        bar,_ = torch.max(bar,1)
        correct_account += torch.sum(bar)
        t.show_loss(show_type=1)
        t.accumulate()
    t.dl.bifurcate(outputs_list)
    print('------use %d -----correct_rate=%5.2f%%---'%(2**bintree_deep,correct_account/600.0))
