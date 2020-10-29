import mnist_web
import numpy as np
import random
import sys
from cradle import Cradle
import torch
import time
from tqdm import tqdm
import my_dataset 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

IF_WANDB = 1
IF_SAVE = 1
LAYER_UNITS = 2000
LAYERS = 3 
CLASS = 10
BATCH_SIZE = 100
NAME = 'neural_400_100'
WORKERS = 15

if IF_WANDB:
    import wandb
    wandb.init(project = 'neural', name = NAME)

dataset = my_dataset.MyDataset(train = True, margin = 3, noise_rate = 0.05)
dataset_test = my_dataset.MyDataset(train = False)
data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()

class Layer():
    def __init__(self, hidden_units = 400):
        self.hidden_units = hidden_units
        self.bits_w = torch.zeros((hidden_units, 5)).cuda()
        self.columns = torch.zeros((hidden_units, 5), dtype=torch.long).cuda()
        self.append_idx = 0

    def append(self, five_columns, five_bits_w):
        if self.append_idx >= self.hidden_units:
            raise Exception("Sorry, append_idx >= hidden_units")
        self.bits_w[self.append_idx] = five_bits_w
        self.columns[self.append_idx] = five_columns
        self.append_idx += 1
        
    def forward(self, inputs):
        data = inputs[:,self.columns[:self.append_idx]]
        data *= self.bits_w[:self.append_idx]
        data = data.sum(2)
        data = (data > 0).float()
        data = data * 2 -1 
        return data

    def load(self, name):
        d = torch.from_numpy(np.load(name + '_data.npy')).cuda()
        print('Load %s Done ..'%(name))
        self.columns = d[:,:,0].type(torch.int64)
        self.bits_w = d[:,:,1].float()
        self.append_idx = self.columns.shape[0] - 1

    def save(self, name):
        d = torch.zeros((self.hidden_units, 5, 2))
        d[:,:,0] = self.columns
        d[:,:,1] = self.bits_w
        d = d.cpu().numpy()
        np.save(name + '_data.npy', d)
        print('Save %s Done ..'%(name))



class CocktailNet(nn.Module):
    def __init__(self, accumulate_features_n, base_size, hid=400):
        super(CocktailNet, self).__init__()
        features_n = accumulate_features_n + 1
        K = (features_n + 1) ** 0.5
        self.weight_1 = torch.randn((base_size, features_n, hid)) / K
        self.bias_1   = torch.randn((base_size, 1, hid)) / K
        self.norm1 = nn.BatchNorm1d(base_size * hid)

        K = (hid * 0.5 + 1) ** 0.5
        self.weight_2 = torch.randn((base_size, hid, hid)) / K
        self.bias_2   = torch.randn((base_size, 1, hid)) / K
        self.norm2 = nn.BatchNorm1d(base_size * hid)

        self.weight_3 = torch.randn((base_size, hid, hid)) / K
        self.bias_3   = torch.randn((base_size, 1, hid)) / K
        self.norm3 = nn.BatchNorm1d(base_size * hid)

        self.weight_4 = torch.randn((base_size, hid, CLASS)) / K
        self.bias_4   = torch.randn((base_size, 1, CLASS)) / K

        self.weight_1 = torch.nn.Parameter(self.weight_1)
        self.weight_2 = torch.nn.Parameter(self.weight_2)
        self.weight_3 = torch.nn.Parameter(self.weight_3)
        self.weight_4 = torch.nn.Parameter(self.weight_4)
        self.bias_1= torch.nn.Parameter(self.bias_1)
        self.bias_2= torch.nn.Parameter(self.bias_2)
        self.bias_3= torch.nn.Parameter(self.bias_3)
        self.bias_4= torch.nn.Parameter(self.bias_4)

    def _apply_batch_norm(self, x, norm_layer):
        base_size = x.shape[0]
        batch_size = x.shape[1]
        x = x.permute(1,0,2)
        x = x.reshape(batch_size, -1)
        x = norm_layer(x)
        x = x.reshape(batch_size, base_size, -1)
        x = x.permute(1,0,2)
        return x

    def forward(self, accumulate_features, base):
        base = base.unsqueeze(-1)
        relu = nn.ReLU(inplace = True)
        x = accumulate_features.unsqueeze(0).repeat(base.shape[0],1,1)
        x = torch.cat((x, base), -1)
        x = x.bmm(self.weight_1)
        x = x + self.bias_1
        x = self._apply_batch_norm(x, self.norm1)
        x = relu(x)

        x = x.bmm(self.weight_2)
        x = x + self.bias_2
        x = self._apply_batch_norm(x, self.norm2)
        x = relu(x)
       
        x = x.bmm(self.weight_3)
        x = x + self.bias_3
        x = self._apply_batch_norm(x, self.norm3)
        x = relu(x)

        x = x.bmm(self.weight_4)
        x = x + self.bias_4
        return x

class CocktailTaster():
    def __init__(self, glide_window = 20):
        self.loss_memory = torch.zeros((1, glide_window)).cuda()
        self.accurate_memory = torch.zeros((1, glide_window)).cuda()
        self.idx = 0
        self.glide_window = glide_window
        pass

    def sip(self, x, labels):
        base_accurate = (x.argmax(-1) == labels.argmax(-1)).float().mean(-1) * 100
        x = x.exp()
        x = x / x.sum(-1).unsqueeze(-1)
        x = -x.log()
        base_loss = (x * labels).sum(-1).mean(-1)
        if base_accurate.shape[0] != self.loss_memory.shape[0]:
            l = base_accurate.shape[0]
            self.loss_memory = self.loss_memory.repeat(l,1)
            self.accurate_memory = self.accurate_memory.repeat(l,1)
        self.loss_memory[:, self.idx] = base_loss
        self.accurate_memory[:, self.idx] = base_accurate 
        self.idx = (self.idx + 1) % self.glide_window
        return base_loss.mean()

    def ban(self, idx):
        self.loss_memory[idx] += 999
        self.accurate_memory[idx] *= 0

    def reveal(self):
        base_mean_accurate = self.accurate_memory.mean(-1)
        base_mean_loss = self.loss_memory.mean(-1)
        return base_mean_loss.argmin(), base_mean_loss.min(), base_mean_accurate.max()

    def gargle(self):
        self.loss_memory = torch.zeros((1, self.glide_window)).cuda()
        self.accurate_memory = torch.zeros((1, self.glide_window)).cuda()
        self.idx = 0


class Drillmaster():
    def __init__(self, layers = [], glide_window = 1000, hid = 100):
        self.layers = layers
        self.reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
        self.reflect_bits_w = torch.zeros((5)).cuda()
        self.f_index = 0
        self.avoid_repeat = []
        self.glide_window = glide_window
        self._init_net()

    def _init_net(self):
        self.cocktailnet = CocktailNet(self.layers[-1].append_idx, 784*2, 100).cuda()
        self.taster = CocktailTaster(self.glide_window)
        self.optimizer = optim.Adam(self.cocktailnet.parameters())

    def add_layer(self, layer):
        self.layers.append(layer)

    def save_last_layer(self, name):
        self.layers[-1].save(name)

    def forward(self, inputs):
        base = inputs
        for i in range(len(self.layers) - 1):
            data = self.layers[i].forward(base)
            base = torch.cat((base, data), 1)
        features = self.layers[-1].forward(base)
        return features, base

    def _get_features_and_base(self, inputs, labels):
        features, base = self.forward(inputs)
        base = base.unsqueeze(-1).repeat(1,1,2)
        base[:,:,1] *= -1
        base = base.reshape((base.shape[0], -1))
        reflection = (base[:,2*self.reflect_columns] * self.reflect_bits_w).sum(1)
        foo = base.t() + reflection
        base = (foo > 0).float() - (foo < 0).float()
        return features, base


    def dissecting_column(self, inputs, labels):
        features, base = self._get_features_and_base(inputs, labels)
        x = self.cocktailnet(features, base)
        loss = self.taster.sip(x, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
    def dissecting_confirm(self):
        for avoid_column in self.avoid_repeat:
            self.taster.ban(avoid_column * 2)
            self.taster.ban(avoid_column * 2 + 1)
        best_idx ,best_loss, best_accurate = self.taster.reveal()
        column = best_idx // 2
        bit_w = (best_idx % 2) * (-2) + 1
        self.reflect_columns[self.f_index] = column
        self.reflect_bits_w[self.f_index] = bit_w
        self.f_index += 1
        self.avoid_repeat.append(column)
        print('%5d    bit_w %2d    loss %8.5f accurate   %8.3f'%\
                (column, bit_w, best_loss,best_accurate))
        if self.f_index == 5:
            self.layers[-1].append(self.reflect_columns, self.reflect_bits_w)
            self.reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
            self.reflect_bits_w = torch.zeros((5)).cuda()
            self.f_index = 0
            self.avoid_repeat = []
            if IF_WANDB:
                wandb.log({'loss':best_loss})
                wandb.log({'acc':best_accurate})
            if IF_SAVE:
                self.save_last_layer(NAME+'_LAYER%d'%0)
        self._init_net()


master = Drillmaster([Layer(200)], 1000, 100)
for hid in range(200):
    print('\n%5d'%hid)
    for f in range(5):
        for i in range(4000):
            images, labels = data_feeder.feed()
            master.dissecting_column(images, labels)
            #if i % 1000 == 999:
            #    master.taster.reveal()
        master.dissecting_confirm()

