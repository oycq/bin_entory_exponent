import mnist_web
import select
import numpy as np
import random
import sys
from dataloader import DataLoader
from cradle import Cradle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm

IF_SAVE = 0
SAVE_NAME = 'five_fc'
ITERATION = 400
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CLASS = 10

dl = DataLoader(True,cuda = 1)
images,labels = dl.get_all()
#images, labels = images[:18000],labels[:18000]
dl_test = DataLoader(False,cuda = 1)
images_t,labels_t = dl_test.get_all()

accumulate = torch.ones((labels.shape[0],CLASS),dtype = torch.float32).cuda()
accumulate_t = torch.ones((labels_t.shape[0],CLASS),dtype = torch.float32).cuda()


saved_data = np.zeros((ITERATION,5,2),dtype = int)
saved_mask = np.zeros((ITERATION,CLASS),dtype = int)

FC_TRAINING_ITERATION = 5000
FC_START_TIME = 1


def get_layer_output(inputs, data):
    aide = inputs[:,:data.shape[0]].clone()
    for i in range(data.shape[0]):
        out_sum = 0
        for j in range(5):
            out_sum += inputs[:,data[i, j, 0]] * data[i,j,1]
        result = (out_sum > 0).float()
        result = result * 2 - 1
        aide[:,i] = result
    return torch.cat([inputs,aide],1)

#images = get_layer_output(images, np.load('direct_v1_data.npy')[:156])
#images_t = get_layer_output(images_t, np.load('direct_v1_data.npy')[:156])


class Enhacne(nn.Module):
    def __init__(self):
        super(Enhacne, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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



def get_loss(o, labels, accum, pretrained_mask = None, enable_fc = False):
    if pretrained_mask is None:
        mask = []
        for i in range(CLASS):
            best = {'j':0, 'loss':9999}
            for j in [0,1,-1]:
                score_table = accum.clone()
                score_table[:,i] += ((j * o) > 0).float()
                if enable_fc:
                    score_table = model(score_table)
                score_table = torch.exp(score_table)
                score_table = score_table / score_table.sum(1).unsqueeze(1)
                loss = -torch.sum(torch.log(score_table) * labels,1)
                loss = loss.mean()
                if loss < best['loss']:
                    best['loss'] = loss
                    best['j'] = j
            mask.append(best['j'])
    else:
        mask = pretrained_mask

    score_table = accum.clone() 
    for i in range(CLASS):
        score_table[:,i] += ((mask[i] * o) > 0).float()
    ret_score_table = score_table.clone()
    if enable_fc:
        score_table = model(score_table)
    score_table = torch.exp(score_table)
    score_table = score_table / score_table.sum(1).unsqueeze(1)
    loss = -torch.sum(torch.log(score_table) * labels,1)
    loss = loss.mean()
    return {'loss':loss,'score_table':ret_score_table,'mask':np.asarray(mask, dtype=np.int8)}

def show_gather(o, labels, mask):
    r = o.unsqueeze(0).mm(labels)
    r = (r + labels.sum(0).unsqueeze(0)) / 2
    output_str = ''
    for i in range(10):
       output_str += '%7d'%(r[0][i])
    print(output_str)
    output_str = ''
    for i in range(10):
       output_str += '%7d'%(mask[i])
    print(output_str)


def show_accuarcate(r, labels, flag = 'train'):#r:classfication_score_table
    a = torch.argmax(r,1)
    b = labels.argmax(1)
    accuarcate = torch.mean((a==b).float())*100
    if flag == 'train':
        print('Train accuarcate:%6.2f%%'%(accuarcate))
    if flag == 'test':
        print('Test accuarcate:%6.2f%%'%(accuarcate))
    if flag == 'silent':
        pass
    return accuarcate


if __name__ == '__main__':
    for iteration in range(ITERATION):
        print('iteration:\n%5d'%iteration)
        
        out_accumu = 0
        avoid_repeat_list = []
        for f in range(1):
            best = {'column':0, 'bit_w':0, 'loss':9999, 'o':None, \
                    'score_table':None, 'mask':None}
            for column in tqdm(range(images.shape[1]), leave=False):
                if column in avoid_repeat_list:#avoid_repeat_list:
                    continue
                for bit_w in [-1,1]:
                    o = out_accumu + bit_w * images[:,column]
                    o[o>0] = 1
                    o[o==0] = 0
                    o[o<0] = -1
                    r = get_loss(o, labels, accumulate, enable_fc=iteration>FC_START_TIME)
                    if r['loss'] < best['loss']:
                        best['column'] = column
                        best['bit_w'] = bit_w
                        best['o']  = o
                        best['loss'] = r['loss']
                        best['mask'] = r['mask']
                        best['score_table'] = r['score_table']
            saved_data[iteration,f,0] = best['column']
            saved_data[iteration,f,1] = best['bit_w']
            saved_mask[iteration] = best['mask']
            print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                    (f,best['column'],best['bit_w'],best['loss']))
            out_accumu += images[:,best['column']] * best['bit_w']
            avoid_repeat_list.append(best['column'])

        show_gather(best['o'],labels,best['mask'])
        show_accuarcate(best['score_table'], labels, 'train')

        test_o = 0
        for i in range(5):
            test_o += images_t[:,saved_data[iteration, i, 0]] * saved_data[iteration,i,1]
        test_o = (test_o > 0).float()
        test_o = test_o * 2 - 1 
        r = get_loss(test_o, labels_t, accumulate_t, best['mask'],iteration>FC_START_TIME)
        print(r['score_table'][:20])
        show_accuarcate(r['score_table'], labels_t, 'test')
        accumulate_t = r['score_table']

        if iteration >= FC_START_TIME-1:
            model = Enhacne().cuda()
            optimizer = optim.Adam(model.parameters())
            for column in tqdm(range(FC_TRAINING_ITERATION), leave=False):
                r = get_loss(best['o'], labels,accumulate, best['mask'], 1)
                loss = r['loss']
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                acc = show_accuarcate(r['score_table'], labels, 'silent')
            print('train %8.4f %9.4f'%(loss,acc))


        accumulate = best['score_table']
        print('\n\n')
        if IF_SAVE:
            np.save(SAVE_NAME + '_data.npy',saved_data)
            np.save(SAVE_NAME + '_mask.npy',saved_mask)




