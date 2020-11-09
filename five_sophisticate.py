import mnist_web
import select
import numpy as np
import random
import sys
from dataloader import DataLoader
from cradle import Cradle
import torch
import time
from tqdm import tqdm
import my_dataset 

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

IF_WANDB = 1
IF_SAVE = 1
LAYER_UNITS = 2000
LAYERS = 3 
CLASS = 10
BATCH_SIZE = 3000
LAYER_NAME = 'new_sophisticate_layer_2000_0.05'
CONSISTENT_THRESH = 5
ALTER_RATE_THRESH = 0.99
WORKERS = 15
EXP_K = 0.05

if IF_WANDB:
    import wandb
    wandb.init()

dataset = my_dataset.MyDataset(train = True, margin = 3, noise_rate = 0.05)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()

class Layer():
    def __init__(self, hidden_units = 400):
        self.hidden_units = hidden_units
        self.bits_w = torch.zeros((hidden_units, 5)).cuda()
        self.columns = torch.zeros((hidden_units, 5), dtype=torch.long).cuda()
        self.masks = torch.zeros((hidden_units, CLASS)).cuda()
        self.append_idx = 0

    def append(self, five_columns, five_bits_w, mask):
        if self.append_idx >= self.hidden_units:
            raise Exception("Sorry, append_idx >= hidden_units")
        self.bits_w[self.append_idx] = five_bits_w
        self.columns[self.append_idx] = five_columns
        self.masks[self.append_idx] = mask
        self.append_idx += 1
        
    def forward(self, inputs):
        data = inputs[:,self.columns]
        data *= self.bits_w
        data = data.sum(2)
        data = (data > 0).float()
        data = data * 2 -1 
        return data

    def get_accumulate(self, inputs):
       data = self.forward(inputs)
       data = data.unsqueeze(2).repeat(1, 1, 10)
       data *= self.masks
       data = (data > 0).float()
       data = data.sum(1) + 1
       return data

    def load(self, name):
        d = torch.from_numpy(np.load(name + '_data.npy')).cuda()
        m = torch.from_numpy(np.load(name + '_mask.npy')).cuda()
        print('Load %s Done ..'%(name))
        self.columns = d[:,:,0].type(torch.int64)
        self.bits_w = d[:,:,1].float()
        self.masks = m.float()

    def save(self, name):
        d = torch.zeros((self.hidden_units, 5, 2))
        d[:,:,0] = self.columns
        d[:,:,1] = self.bits_w
        d = d.cpu().numpy()
        m = self.masks.cpu().numpy()
        np.save(name + '_data.npy', d)
        np.save(name + '_mask.npy', m)
        print('Save %s Done ..'%(name))
        

class Drillmaster():
    def __init__(self, layers = []):
        self.layers = layers
        self.excitation = torch.eye(CLASS,CLASS).unsqueeze(1).cuda()
        self.reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
        self.reflect_bits_w = torch.zeros((5)).cuda()
        self.current_loss = 0
        self.base_influence = 0
        self.base_loss = 0
        self.previous_mask = 0
        self.f_index = 0
        self.avoid_repeat = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def save_last_layer(self, name):
        self.layers[-1].save(name)

    def forward(self, inputs):
        base = inputs
        for i in range(len(self.layers) - 1):
            data = self.layers[i].forward(base)
            base = torch.cat((base, data), 1)
        accumulate = self.layers[-1].get_accumulate(base)
        return accumulate, base

    def _crorss_entropy(self, score_table, labels, exp_k=EXP_K):
        a = score_table - score_table.mean(-1).unsqueeze(-1)
        entropy = torch.exp(a * exp_k)
        entropy = entropy / entropy.sum(-1).unsqueeze(-1)
        entropy = -torch.log(entropy)
        cross_entropy = (entropy * labels).sum(-1)
        return cross_entropy

    def get_accurate(self, inputs, labels):
        r,_ = self.forward(inputs)
        a = r.argmax(1)
        b = labels.argmax(1)
        accuarcate = torch.mean((a==b).float())*100
        return accuarcate

    def _get_acucumulate_and_base(self, inputs, labels):
        accumulate, base = self.forward(inputs)
        base = base.unsqueeze(-1).repeat(1,1,2)
        base[:,:,1] *= -1
        base = base.reshape((base.shape[0], -1))
        reflection = (base[:,2*self.reflect_columns] * self.reflect_bits_w).sum(1)
        foo = base.t() + reflection
        base = (foo > 0).float() - (foo < 0).float()
        return accumulate, base

    def carving_mask(self, inputs, labels):
        accumulate, base = self._get_acucumulate_and_base(inputs, labels)
        new_accumulate = self.excitation + accumulate 
        loss_ori = self._crorss_entropy(accumulate, labels)
        loss_new = self._crorss_entropy(new_accumulate, labels).t()
        loss_delta = loss_ori.unsqueeze(1) - loss_new
        loss_delta_sum = loss_delta.sum(0)

        base_positive_influence = base.mm(loss_delta)
        base_negtive_influence = loss_delta_sum - base_positive_influence
        base_no_influence = torch.zeros_like(base_positive_influence)
        self.base_influence += torch.cat((base_negtive_influence.unsqueeze(-1),\
                               base_no_influence.unsqueeze(-1),\
                               base_positive_influence.unsqueeze(-1)), 2) 
        mask = torch.argmax(self.base_influence, -1) - 1
        alter_rate = 1 - ((mask - self.previous_mask).float() / 2).abs().mean()
        self.previous_mask = mask
        return alter_rate

    def dissecting_column(self, inputs, labels):
        accumulate, base = self._get_acucumulate_and_base(inputs, labels)
        base_mask = torch.argmax(self.base_influence, -1) - 1
        base_mask = base_mask.float()
        new_accumulate = (base.unsqueeze(-1).bmm(
                        base_mask.unsqueeze(1)) > 0).float() + accumulate
        current_base_loss = self._crorss_entropy(new_accumulate, labels).mean(-1) 
        self.base_loss += current_base_loss
        for avoid_column in self.avoid_repeat:
            bar = avoid_column * 2
            self.base_loss[bar:bar+2] *= 0
            self.base_loss[bar:bar+2] += 99
        best_index = self.base_loss.argmin()
        self.current_loss = current_base_loss[best_index]
        column = best_index // 2
        mask = base_mask[best_index]
        bit_w = (best_index % 2) * (-2) + 1
        return column.item()

    def dissecting_confirm(self, hook_time = 0):
        base_mask = torch.argmax(self.base_influence, -1) - 1
        best_index = self.base_loss.argmin()
        loss = self.current_loss
        column = best_index // 2
        self.avoid_repeat.append(column)
        bit_w = (best_index % 2) * (-2) + 1
        mask = base_mask[best_index]
        self.reflect_columns[self.f_index] = column
        self.reflect_bits_w[self.f_index] = bit_w
        self.base_influence = 0
        self.base_loss = 0
        self.previous_mask = 0
        self.f_index += 1
        print('%5d  %2d  %8.5f %5d'%(column, bit_w, loss, hook_time))
        if self.f_index == 5:
            f_index = 0
            self.layers[-1].append(self.reflect_columns, \
            self.reflect_bits_w, mask)
            self.reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
            self.reflect_bits_w = torch.zeros((5)).cuda()
            self.f_index = 0
            self.avoid_repeat = []
        

drillmaster = Drillmaster([Layer(LAYER_UNITS)])
for j in range(LAYERS):
    for l in range(LAYER_UNITS):
        print('l%du%d'%(j+1,l+1))
        t1 = time.time() * 1000
        for k in range(5):
            hook_time = 0
            while(1):
                hook_time += 1
                images, labels = data_feeder.feed()
                alter_rate = drillmaster.carving_mask(images, labels)
                if alter_rate > ALTER_RATE_THRESH:
                    break
            previous_column = 0
            consistent = 0
            while(1):
                hook_time += 1
                images, labels = data_feeder.feed()
                alter_rate = drillmaster.carving_mask(images, labels)
                column = drillmaster.dissecting_column(images, labels)
                if previous_column == column:
                    consistent += 1
                else:
                    consistent = 0
                previous_column = column
                if consistent >= CONSISTENT_THRESH:
                    break
            drillmaster.dissecting_confirm(hook_time)
        train_accurate = drillmaster.get_accurate(images, labels)
        test_accurate = drillmaster.get_accurate(images_t, labels_t)
        if IF_WANDB:
            wandb.log({'train':train_accurate})
            wandb.log({'test':test_accurate})
        print('Train accurate =%8.3f%%'%(train_accurate))
        print('Test  accurate =%8.3f%%'%(test_accurate))
        t2 = time.time() * 1000
        print('Caculate  time =%7dms\n'%(t2-t1))
    if IF_SAVE:
        drillmaster.save_last_layer(LAYER_NAME+'%d'%j)
    drillmaster.add_layer(Layer(LAYER_UNITS))

