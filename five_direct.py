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

IF_SAVE = 0
SAVE_NAME = 'five_direct_l3'
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

images = get_layer_output(images, np.load('five_direct_v1_data.npy')[:])
images_t = get_layer_output(images_t, np.load('five_direct_v1_data.npy')[:])
images = get_layer_output(images, np.load('five_direct_l2_data.npy')[:])
images_t = get_layer_output(images_t, np.load('five_direct_l2_data.npy')[:])


broadcast_mask = torch.zeros((CLASS, 3 , 1, CLASS)).cuda()
for i in range(10):
    for j in range(3):
        broadcast_mask[i,j,0,i] = j - 1

def get_loss(o, labels, accum, pretrained_mask = None, exp_k = 0.25):
    if pretrained_mask is None:
        broadcast = (o.unsqueeze(1).matmul(broadcast_mask) > 0).float()
        broadcast += accum.unsqueeze(0).unsqueeze(0)
        loss = torch.exp(broadcast * exp_k)
        loss = loss / loss.sum(3).unsqueeze(3)
        loss = -torch.sum(torch.log(loss) * labels,3)
        loss = loss.mean(2)
        mask = loss.argmin(1).float() - 1
    else:
        mask = pretrained_mask

    score_table = accum.clone() 
    score_table += (o.unsqueeze(1).mm(mask.unsqueeze(0)) > 0).float()
    loss = torch.exp(score_table * exp_k)
    loss = loss / loss.sum(1).unsqueeze(1)
    loss = -torch.sum(torch.log(loss) * labels,1)
    loss = loss.mean()
    return {'loss':loss,'score_table':score_table,'mask':mask}


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


def show_accuarcate(r, labels, train=True):#r:classfication_score_table
    a = torch.argmax(r,1)
    b = labels.argmax(1)
    accuarcate = torch.mean((a==b).float())*100
    if train:
        print('Train accuarcate:%6.2f%%'%(accuarcate))
    else:
        print('Test accuarcate:%6.2f%%'%(accuarcate))
    return accuarcate


if __name__ == '__main__':
    for iteration in range(ITERATION):
        print('iteration:\n%5d'%iteration)
        out_accumu = 0
        avoid_repeat_list = []
        for f in range(5):
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
                    r = get_loss(o, labels, accumulate)
                    if r['loss'] < best['loss']:
                        best['column'] = column
                        best['bit_w'] = bit_w
                        best['o']  = o
                        best['loss'] = r['loss']
                        best['mask'] = r['mask']
                        best['score_table'] = r['score_table']
            saved_data[iteration,f,0] = best['column']
            saved_data[iteration,f,1] = best['bit_w']
            saved_mask[iteration] = best['mask'].cpu()
            print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                    (f,best['column'],best['bit_w'],best['loss']))
            out_accumu += images[:,best['column']] * best['bit_w']
            avoid_repeat_list.append(best['column'])

        show_gather(best['o'],labels,best['mask'])
        show_accuarcate(best['score_table'], labels)
        accumulate = best['score_table']

        test_o = 0
        for i in range(5):
            test_o += images_t[:,saved_data[iteration, i, 0]] * saved_data[iteration,i,1]
        test_o = (test_o > 0).float()
        test_o = test_o * 2 - 1 
        r = get_loss(test_o, labels_t, accumulate_t, best['mask'])
        print(r['score_table'][:20])
        show_accuarcate(r['score_table'], labels_t)
        accumulate_t = r['score_table']
        print('\n\n')
        if IF_SAVE:
            np.save(SAVE_NAME + '_data.npy',saved_data)
            np.save(SAVE_NAME + '_mask.npy',saved_mask)




