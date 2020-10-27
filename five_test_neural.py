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

IF_WANDB = 0
IF_SAVE = 1
LAYER_UNITS = 2000
LAYERS = 3 
CLASS = 10
BATCH_SIZE = 100
LAYER_NAME = 'neural_test1'
WORKERS = 15

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


class Drillmaster():
    def __init__(self, layers = []):
        self.layers = layers
        self.reflect_columns = torch.zeros((5), dtype=torch.long).cuda()
        self.reflect_bits_w = torch.zeros((5)).cuda()
        self.current_loss = 0
        self.base_influence = 0
        self.base_loss = 0
        self.f_index = 0
        self.avoid_repeat = []
        self.cocktailnet = CocktailNet(0, 784*2).cuda()

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
        print(x.shape)
        sys.exit(0)
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
        base = base.t().unsqueeze(-1)
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
        #print(x.min().item(), x.max().item(), x.mean().item(), x.std().item())
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

    def reveal(self):
        base_mean_accurate = self.accurate_memory.mean(-1)
        base_mean_loss = self.loss_memory.mean(-1)
        print('accurate  best %8.4f   worse  %8.4f   idx   %5d'%(base_mean_accurate.max(),\
                    base_mean_accurate.min(),base_mean_accurate.argmax()%784))
        print('loss      best %8.4f   worse  %8.4f   idx   %5d'%(base_mean_loss.min(),\
                    base_mean_loss.max(), base_mean_loss.argmin()%784))

    def gargle(self):
        self.loss_memory = torch.zeros((1, self.glide_window)).cuda()
        self.accurate_memory = torch.zeros((1, self.glide_window)).cuda()
        self.idx = 0




select_features = torch.LongTensor([541, 488, 345, 542, 317, 467]).cuda()
cocktail = CocktailNet(select_features.shape[0], 784*2, 100).cuda()
optimizer = optim.Adam(cocktail.parameters())
taster = CocktailTaster(200)
for i in range(10000):
    images, labels = data_feeder.feed()
    base = torch.cat((images, images * -1), -1)
    accumulate_features = images[:, select_features]
    x = cocktail(accumulate_features, base)
    loss = taster.sip(x, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 200 == 0:
        taster.reveal()

sys.exit(0)

for i in range(2000):
    images, labels = data_feeder.feed()
    optimizer.zero_grad()
    output = net(images)
    loss = loss_function(output, labels)
    loss.backward()
    optimizer.step()
    output_t = net(images_t)

    if i % 10 == 0:
        print('%d'%i)
        print('train')
        get_accurate(output, labels)
        print('test')
        get_accurate(output_t, labels_t)
        print(' ')

sys.exit(0)
mas = Drillmaster([Layer(100)])
a = mas._get_features_and_base(images_t, labels_t)
print(a.shape)
        
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

