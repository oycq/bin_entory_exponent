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
BATCH_SIZE = 5000
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
        features, base = self._get_acucumulate_and_base(inputs, labels)
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


class Net(nn.Module):
    def __init__(self, inputs_size, hidden=400):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inputs_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace = True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace = True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace = True),
            nn.Linear(hidden, CLASS),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for i,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                   nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Net1(nn.Module):
    def __init__(self, in_dim, base_size,  hid=400):
        super(Net1, self).__init__()
        self.linear1_w = torch.nn.Parameter(torch.zeros(base_size, in_dim, hid))
        self.linear1_bias = torch.nn.Parameter(torch.ones(base_size, hid))
        self.batchnorm1 = torch.nn.BatchNorm1d(base_size, hid)

        self.linear2_w = torch.nn.Parameter(torch.zeros(base_size, hid, hid))
        self.linear2_bias = torch.nn.Parameter(torch.ones(base_size, hid))
        self.batchnorm2 = torch.nn.BatchNorm1d(base_size, hid)

        self.linear3_w = torch.nn.Parameter(torch.zeros(base_size, hid, hid))
        self.linear3_bias = torch.nn.Parameter(torch.ones(base_size, hid))
        self.batchnorm3 = torch.nn.BatchNorm1d(base_size, hid)

        self.linear4_w = torch.nn.Parameter(torch.zeros(base_size, hid, CLASS))
        self.linear4_bias = torch.nn.Parameter(torch.ones(base_size, CLASS))
        #nn.ReLU(inplace = True),

    def forward(self, x):
        x = x.bmm(self.linear1_w)
        x = x.bmm(self.linear2_w)
        x = x.bmm(self.linear3_w)
        x = x.bmm(self.linear4_w)
        return x


net = Net1(784,100,200).cuda().train()
optimizer = optim.Adam(net.parameters())
loss_function = nn.CrossEntropyLoss()

def get_accurate(outputs, labels):
    predict = outputs.argmax(1)
    accurate = torch.mean((predict==labels).float())*100
    print(accurate)


images, labels = data_feeder.feed()
images = images.unsqueeze(0).repeat(100,1,1)
labels = labels.unsqueeze(0).repeat(100,1,1)
labels = labels.reshape(-1)
time1 = time.time()
for i in range(200):
    optimizer.zero_grad()
    output = net(images).reshape(-1,10)
    loss = loss_function(output, labels)
    loss.backward()
    optimizer.step()

    print('%d'%i)
    #    print('train')
    #    get_accurate(output, labels)
    #    print('test')
    #    get_accurate(output_t, labels_t)
    #    print(' ')
time2 = time.time()
print(time2-time1)
sys.exit(0)


class Cradle():
    def __init__(self, inputs_size, base_size, hidden=400):
        self.base_size = base_size
        self.nets = []
        self.optimizers = []
        self.loss_function = nn.CrossEntropyLoss()
        for i in range(self.base_size):
            net = Net(inputs_size, hidden).train()
            optimizer = optim.Adam(net.parameters())
            self.nets.append(net)
            self.optimizers.append(optimizer)
        
    def train(self, inputs, labels):
        a = 0
        for net, optimizer in zip(self.nets, self.optimizers):
            print(a)
            a = a + 1
            optimizer.zero_grad()
            net = net.cuda()
            output = net(inputs)
            loss = self.loss_function(output, labels)
            loss.backward()
            net = net.cpu()
            optimizer.step()


def get_accurate(outputs, labels):
    predict = outputs.argmax(1)
    accurate = torch.mean((predict==labels).float())*100
    print(accurate)

#cradle = Cradle(784, 1500, 400)
#for i in range(200):
#    images, labels = data_feeder.feed()
#    t1 = time.time()
#    cradle.train(images, labels)
#    t2 = time.time()
#    cradle.nets[0] =  cradle.nets[0].cuda()
#    output = cradle.nets[0](images)
#    output_t = cradle.nets[0](images_t)
#    cradle.nets[0] =  cradle.nets[0].cpu()
#    print('%d  %7.1f'%(i, t2-t1))
#    print('train')
#    get_accurate(output, labels)
#    print('test')
#    get_accurate(output_t, labels_t)
#    print(' ')
#
#
#
#sys.exit(0)
net = Net(784).cuda().train()
optimizer = optim.Adam(net.parameters())
loss_function = nn.CrossEntropyLoss()




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

