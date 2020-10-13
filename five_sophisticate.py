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

IF_SAVE = 0
LAYER_UNITS = 400
LAYERS = 5
CLASS = 10
BATCH_SIZE = 10000
BASE_SPLIT = 50
LAYER_NAME = 'five_sophisticate_layer'


dataset = my_dataset.MyDataset(train = True, margin = 3, noise_rate = 0.05)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = 0)
images_t,labels_t = dataset_test.get_all()

#dataset = my_dataset.MyDataset(train = True, margin = 1, noise_rate = 0.05)
#data_feeder = my_dataset.DataFeeder(dataset, 60000, num_workers = 0)
#images_t,labels_t = data_feeder.feed()

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
        


class Preprocess():
    def __init__(self, layers = []):
        self.layers = layers
        self.excitation = torch.eye(CLASS,CLASS).unsqueeze(1).cuda()


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

    def _crorss_entropy(self, score_table, labels, exp_k=0.25):
        entropy = torch.exp(score_table * exp_k)
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

    def refine(self, inputs, labels):
        accumulate, whole_base = self.forward(inputs)
        reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
        reflect_bits_w = torch.zeros((5)).cuda()
        avoid_repeat = []
        for f in range(5):
            reflection = (whole_base[:,reflect_columns] * reflect_bits_w).sum(1)
            best = {'column':None, 'mask':None, 'loss':99999}
            for bit_w in [-1,1]:
                foo = whole_base.t() * bit_w + reflection
                equivalent_whole_base = (foo > 0).float() - (foo < 0).float()
                split_bias = -BASE_SPLIT
                for base in torch.split(equivalent_whole_base, BASE_SPLIT, dim=0):
                    split_bias += BASE_SPLIT
                    loss_ori = self._crorss_entropy(accumulate, labels)
                    new_accumulate = self.excitation + accumulate 
                    loss_new = self._crorss_entropy(new_accumulate, labels).t()
                    loss_delta = loss_ori.unsqueeze(1) - loss_new
                    loss_delta_sum = loss_delta.sum(0)
                    base_positive_influence = base.mm(loss_delta)
                    base_negtive_influence = loss_delta_sum - base_positive_influence
                    base_no_influence = torch.zeros_like(base_positive_influence)
                    base_influence = torch.cat((base_negtive_influence.unsqueeze(-1),\
                                           base_no_influence.unsqueeze(-1),\
                                           base_positive_influence.unsqueeze(-1)), 2) 
                    base_mask = torch.argmax(base_influence, -1) - 1
                    base_mask = base_mask.float()
                    accumulate_new = (base.unsqueeze(-1).bmm(
                            base_mask.unsqueeze(1)) > 0).float() + accumulate
                    base_loss = self._crorss_entropy(accumulate_new, labels).mean(-1)
                    best_index = base_loss.argmin()
                    best_loss = base_loss[best_index]
                    column = best_index + split_bias
                    if best_loss < best['loss'] and column not in avoid_repeat:
                        best['loss'] = best_loss
                        best['column'] = best_index + split_bias 
                        best['mask'] = base_mask[best_index]
                        best['bit_w'] = bit_w
                        avoid_repeat.append(column)
            reflect_columns[f] = best['column']
            reflect_bits_w[f] = best['bit_w']
            print(best['column'].item(), best['bit_w'],best['loss'].item())
        self.layers[-1].append(reflect_columns, reflect_bits_w, best['mask'])


preprocess = Preprocess([Layer(LAYER_UNITS)])
images, labels = data_feeder.feed()
for j in range(LAYERS):
    for i in range(LAYER_UNITS):
        t1 = time.time() * 1000
        preprocess.refine(images, labels)
        a = preprocess.get_accurate(images, labels)
        b = preprocess.get_accurate(images_t, labels_t)
        t2 = time.time() * 1000
        print('l%dh%d   %5d'%(j,i,t2-t1))
        print('Train accurate=%8.3f%%'%(a))
        print('Test accurate=%8.3f%%'%(b))
        print('')
    if IF_SAVE:
        preprocess.save_last_layer(LAYER_NAME+'%d'%j)
        preprocess.add_layer(Layer(LAYER_UNITS))










