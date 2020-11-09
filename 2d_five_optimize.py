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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


dl = DataLoader(True,cuda = 1)
images,labels = dl.get_all()
images, labels = images[:18000],labels[:18000]
dl_test = DataLoader(False,cuda = 1)
images_t,labels_t = dl_test.get_all()
images_t, labels_t = images_t,labels_t


def get_classfication_score_table(o,labels):
    r = o.mm( o.t().mm(labels)  )
    r = (r + labels.sum(0).unsqueeze(0)) / 2
    return r

def show_gather(o, labels):
    r = o.t().mm(labels)
    r = (r + labels.sum(0).unsqueeze(0)) / 2
    output_str = ''
    for i in range(10):
       output_str += '%7d'%(r[0][i])
    print(output_str)


def get_loss(class_s_table, labels):
    r = class_s_table / class_s_table.sum(1).unsqueeze(1)
    r = -torch.sum(torch.log(r) * labels,1)
    r = r.mean()
    return r

def show_accuarcate(r, labels, train=True):#r:classfication_score_table
    a = torch.argmax(r,1)
    b = labels.argmax(1)
    accuarcate = torch.mean((a==b).float())*100
    if train:
        print('Train accuarcate:%6.2f%%\n\n'%(accuarcate))
    else:
        print('Test accuarcate:%6.2f%%\n\n'%(accuarcate))
    return accuarcate


#if __name__ == '__main__':
#    for iteration in range(10):
#        print('iteration:\n%5d'%iteration)
#        out_accumu = 0
#        avoid_repeat_list = []
#        for f in range(5):
#            best = {'column':0, 'bit_w':0, 'loss':9999, 'o':None}
#            for column in tqdm(range(images.shape[1]), leave=False):
#                if column in avoid_repeat_list:
#                    continue
#                for bit_w in [-1,1]:
#                    rand_swing = torch.randint(0, 2, (images.shape[0],)).cuda().float()
#                    o = out_accumu + bit_w * images[:,column] + rand_swing
#                    o = (o >= 1).float().unsqueeze(1)
#                    o = o * 2 - 1
#                    r = get_classfication_score_table(o, labels)
#                    r = get_loss(r, labels)
#                    if r < best['loss']:
#                        best['loss'] = r.item()
#                        best['column'] = column
#                        best['bit_w'] = bit_w
#                        best['o']  = o
#            print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
#                    (f,best['column'],best['bit_w'],best['loss']))
#            out_accumu += images[:,best['column']] * best['bit_w']
#            avoid_repeat_list.append(best['column'])
#
#        o = (out_accumu > 0).float().unsqueeze(1)
#        o = o * 2 - 1
#        r = get_classfication_score_table(o,labels)
#        show_gather(best['o'],labels)
#        show_accuarcate(r, labels)
#        sys.exit(0)




if __name__ == '__main__':
    for iteration in range(10):
        print('iteration:\n%5d'%iteration)
        out_accumu = 0
        avoid_repeat_list = []
        f = 1
        best = {'column':0, 'bit_w':0, 'loss':9999, 'o':None}
        for column in tqdm(range(images.shape[1]), leave=False):
            if column in avoid_repeat_list:
                continue
            for bit_w in [-1,1]:
                o = out_accumu + bit_w * images[:,column]
                o = (o >= 0).float().unsqueeze(1)
                o = o * 2 - 1
                r = get_classfication_score_table(o, labels)
                r = get_loss(r, labels)
                if r < best['loss']:
                    best['loss'] = r.item()
                    best['column'] = column
                    best['bit_w'] = bit_w
                    best['o']  = o
        print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                (f,best['column'],best['bit_w'],best['loss']))
        out_accumu += images[:,best['column']] * best['bit_w']
        avoid_repeat_list.append(best['column'])

        
        best = {'column1':0, 'bit_w1':0, 'column2':0, 'bit_w2':0, 'loss':9999, 'o':None}
        for column1 in [376]:
            for column2 in [429]:
                if column1 in avoid_repeat_list:
                    continue
                if column2 in avoid_repeat_list:
                    continue
                for bit_w1 in [-1,1]:
                    for bit_w2 in [-1,1]:
                        o = out_accumu + bit_w1 * images[:,column1] + bit_w2 * images[:,column2]
                        o = (o >= 0).float().unsqueeze(1)
                        o = o * 2 - 1
                        r = get_classfication_score_table(o, labels)
                        r = get_loss(r, labels)
                        if r < best['loss']:
                            best['loss'] = r.item()
                            best['column1'] = column1
                            best['bit_w1'] = bit_w1
                            best['column2'] = column2
                            best['bit_w2'] = bit_w2
                            best['o']  = o
        print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                (1,best['column1'],best['bit_w1'],best['loss']))
        print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                (2,best['column2'],best['bit_w2'],best['loss']))

        out_accumu += images[:,best['column1']] * best['bit_w1']
        out_accumu += images[:,best['column2']] * best['bit_w2']
        avoid_repeat_list.append(best['column1'])
        avoid_repeat_list.append(best['column2'])


        best = {'column1':0, 'bit_w1':0, 'column2':0, 'bit_w2':0, 'loss':9999, 'o':None}
        for column1 in tqdm(range(images.shape[1]), leave=False):
            for column2 in tqdm(range(images.shape[1]), leave=False):
                if column1 in avoid_repeat_list:
                    continue
                if column2 in avoid_repeat_list:
                    continue
                for bit_w1 in [-1,1]:
                    for bit_w2 in [-1,1]:
                        o = out_accumu + bit_w1 * images[:,column1] + bit_w2 * images[:,column2]
                        o = (o >= 0).float().unsqueeze(1)
                        o = o * 2 - 1
                        r = get_classfication_score_table(o, labels)
                        r = get_loss(r, labels)
                        if r < best['loss']:
                            best['loss'] = r.item()
                            best['column1'] = column1
                            best['bit_w1'] = bit_w1
                            best['column2'] = column2
                            best['bit_w2'] = bit_w2
                            best['o']  = o
        print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                (3,best['column1'],best['bit_w1'],best['loss']))
        print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                (4,best['column2'],best['bit_w2'],best['loss']))



        out_accumu += images[:,best['column1']] * best['bit_w1']
        out_accumu += images[:,best['column2']] * best['bit_w2']
        avoid_repeat_list.append(best['column1'])
        avoid_repeat_list.append(best['column2'])

        o = (out_accumu > 0).float().unsqueeze(1)
        o = o * 2 - 1
        r = get_classfication_score_table(o,labels)
        show_gather(best['o'],labels)
        show_accuarcate(r, labels)
        sys.exit(0)

#iteration:
#    0
#    0   378   bit_w:-1  loss: 2.08234                                              
#    1   350   bit_w:-1  loss: 2.10095                                             
#    2   429   bit_w: 1  loss: 2.05335                                                  
#    3   405   bit_w:-1  loss: 2.06593                                                
#    4   456   bit_w: 1  loss: 2.03012                                               
#   1699     35   1214    134   1374    472   1279   1695    256    945
#Train accuarcate: 20.64%


#iteration:
#    0
#    1   378   bit_w:-1  loss: 2.08234                                              
#    1   376   bit_w:-1  loss: 2.04831
#    2   429   bit_w: 1  loss: 2.04831
#    3   153   bit_w:-1  loss: 2.02929                                                
#    4   399   bit_w: 1  loss: 2.02929                                             
#   1640     30    730    143   1507    532   1071   1752    244   1236
#Train accuarcate: 20.96%
