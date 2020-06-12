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
        self.entropy_leaves = [999999]
        self.bins_leaves = [None]
        self.name_leaves= ['']
        self.leaf_i = 0
        self.current_split_bins = []
    
    def bifurcate(self,outputs,lr_entropy,bins):
        name = self.name_leaves[self.leaf_i]
        l_entropy = lr_entropy[0].cpu().item()
        r_entropy = lr_entropy[1].cpu().item()
        l_bins = bins[0].cpu()
        r_bins = bins[1].cpu()
        self.current_split_bins = [l_bins+r_bins,l_bins,r_bins]
        images_leaf = self.images_leaves[self.leaf_i]
        labels_leaf = self.labels_leaves[self.leaf_i]
        self.images_leaves.insert(self.leaf_i+1,images_leaf[outputs == 1])
        self.labels_leaves.insert(self.leaf_i+1,labels_leaf[outputs == 1])
        self.bins_leaves.insert(self.leaf_i+1,r_bins)
        self.name_leaves.insert(self.leaf_i+1,name + '1')
        self.entropy_leaves.insert(self.leaf_i+1,r_entropy)
        self.images_leaves.insert(self.leaf_i+1,images_leaf[outputs == 0])
        self.labels_leaves.insert(self.leaf_i+1,labels_leaf[outputs == 0])
        self.bins_leaves.insert(self.leaf_i+1,l_bins)
        self.name_leaves.insert(self.leaf_i+1,name + '0')
        self.entropy_leaves.insert(self.leaf_i+1,l_entropy)
        del self.images_leaves[self.leaf_i]
        del self.labels_leaves[self.leaf_i]
        del self.entropy_leaves[self.leaf_i]
        del self.name_leaves[self.leaf_i]
        del self.bins_leaves[self.leaf_i]
        self.leaf_i = self.entropy_leaves.index(max(self.entropy_leaves))

    def get_leaf(self):
        return self.images_leaves[self.leaf_i],self.labels_leaves[self.leaf_i]

    def print_statue(self):
        correct_count = 0
        total_count = 0
        for i in range(len(self.entropy_leaves)):
            name = self.name_leaves[i]
            e_n = self.images_leaves[i].shape[0]
            e_sum = self.entropy_leaves[i] 
            e = e_sum / e_n
            bins = self.bins_leaves[i]
            correct_count += torch.max(bins)
            total_count += torch.sum(bins)
            bins_string = ''
            for j in range(bins.shape[0]):
                bins_string += '%5d'%bins[j]
            print('== %-10s e_n=%6d   e=%6.3f   sum_j=%8.0f   %s'%\
                    (name,e_n,e,e_sum,bins_string))
        print('')
        for bins in self.current_split_bins:
            bins_string = ''
            for j in range(bins.shape[0]):
                bins_string += '%5d'%bins[j]
            print('%s'%(bins_string))

        print('total correct_rate = %6.2f%%'%(correct_count.item() * 100.0 / total_count.item()))


class Training(): 
    def __init__(self, inputs_n ,cradle_n=50, repro_n = 500, repro_bunch = 20, cuda = False):
        self.cradle = Cradle(cradle_n, inputs_n, mutation_rate = 0.005,
                fading_rate = 0.99995,cuda=cuda)
        self.dl = Bifurcate_loader(train = True, cuda=cuda)
        _, self.prior_labels = self.dl.get_leaf()
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
        
        lr_entropy = -torch.sum(bins* torch.log(bins/ bins_sum),1)
        worse_entroypy=torch.max(lr_entropy/torch.sum(bins,1))

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
                bins2,bins2_entropy,global_entropy,loss_k,lr_entropy
        #print(correct_rate,global_entropy)

    def train_one_bunch(self):
        inputs, labels = self.dl.get_leaf()
        bunch_w = self.cradle.get_w(self.repro_bunch)
        bunch_w = bunch_w.reshape(-1,1,7,7)
        outputs = torch.nn.functional.conv2d(inputs, bunch_w, bias=None,\
                stride=1, padding=0, dilation=1, groups=1)
        outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],-1)
        outputs = (outputs - outputs.mean(2).unsqueeze(-1)) / outputs.std(2).unsqueeze(-1)
        mask = outputs > 1
        outputs[mask] = 1
        outputs[~mask] = 0
        outputs = outputs.sum(2)
        outputs = (outputs - outputs.mean(0)) / outputs.std(0)
        mask = outputs > 0
        outputs[mask] = 1
        outputs[~mask] = 0
        outputs = outputs.type(torch.int32)
        bunch_w = bunch_w.reshape(-1,bunch_w.shape[0])
        new_labels = labels + outputs * 10 * (2 ** self.prior_n)
        bunch_loss = torch.zeros((self.repro_bunch,1))
        if self.cuda:
            bunch_loss = bunch_loss.cuda()
        for i in range(self.repro_bunch):
            label = new_labels[:,i]
            bins = torch.bincount(label,minlength=10 * (2 ** self.prior_n) * 2)
            bunch_loss[i],correct_rate,bins,bins2,bins2_entropy,\
                    global_entropy,loss_k,lr_entropy = self.analyse_bins(bins)
            if bunch_loss[i] < self.best_result['loss']:
                self.best_result['loss'] = bunch_loss[i]
                self.best_result['w'] = bunch_w[:,i]
                self.best_result['label'] = new_labels[:,i].reshape(-1,1)
                self.best_result['correct_rate'] = correct_rate
                self.best_result['bins'] = bins 
                self.best_result['bins2'] = bins2
                self.best_result['global_entropy'] = global_entropy 
                self.best_result['bins2_entropy'] = bins2_entropy 
                self.best_result['loss_k'] = loss_k
                self.best_result['outputs'] = outputs[:,i]
                self.best_result['lr_entropy'] = lr_entropy
        bunch_w = bunch_w.permute(1, 0)
        self.cradle.pk(bunch_w,bunch_loss)

    def reset(self):
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
        if j == 1:
            self.cradle.set_fading_rate(0.99999)
        if j == 3:
            self.cradle.set_fading_rate(0.999995)
        if j == 5:
            self.cradle.set_fading_rate(0.999999)
        if j == 9:
            self.cradle.set_fading_rate(0.9999995)

CRADLE_N = 50
PARAMS = 49 
REPRO_N = 50
REPRO_BUNCH = 5
J = 3
CUDA = 0
LEAVES_N = 128
SAVE_PATH = './maxsume_leaf.npy'

t = Training(inputs_n = PARAMS,cradle_n= CRADLE_N,\
        repro_n = CRADLE_N, repro_bunch = REPRO_BUNCH,cuda=CUDA)
need_save = np.zeros((PARAMS, LEAVES_N),dtype = int)

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
    need_save[:,leaves_n] = t.best_result['w'].cpu()
    np.save(SAVE_PATH, need_save)
    t.reset()
    t.dl.print_statue()
