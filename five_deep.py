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

IF_SAVE = 1
SAVE_NAME = 'five_2layer.npy'
ITERATION = 784 
TOP_K_RATE = 0.005


dl = DataLoader(True,cuda = 1)
images,labels = dl.get_all()
images, labels = images[:10000],labels[:10000]
dl_test = DataLoader(False,cuda = 1)
images_t,labels_t = dl_test.get_all()
images_t, labels_t = images_t,labels_t


accumulate = torch.zeros((labels.shape[0],labels.shape[0]),dtype = torch.float32)
accumulate -= 9999 * torch.eye(labels.shape[0])
accumulate_t = torch.zeros((labels_t.shape[0],labels_t.shape[0]),dtype = torch.float32)
accumulate_t -= 9999 * torch.eye(labels_t.shape[0])

accumulate = accumulate.cuda()
accumulate_t = accumulate_t.cuda()
saved_data = np.zeros((ITERATION,5,2),dtype = int)


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

images = get_layer_output(images, np.load('./five_v1.npy')[:216])
images_t = get_layer_output(images_t, np.load('./five_v1.npy')[:216])


def get_similarity_table(o1,o2):
    r = o1.mm(o2.t())
    return r


def get_classfication_score_table(similar_table, labels, accumulate):
    r = similar_table + accumulate
    top_k_elements,_ = torch.topk(r, int(labels.shape[0]*TOP_K_RATE))
    throat,_ = top_k_elements.min(1)
    throat = throat.unsqueeze(1)
    mask = (r >= throat).float()
    r *= mask
    r = 2 ** r
    r = r.mm(labels)
    r /= labels.sum(0).unsqueeze(0)
    return r

def show_gather(o, labels):
    r = o.t().mm(labels)
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
        #print(r[:5,:9])
        #print(labels[:5])
    else:
        print('Test accuarcate:%6.2f%%\n\n'%(accuarcate))
    return accuarcate


if __name__ == '__main__':
    for iteration in range(ITERATION):
        print('iteration:\n%5d'%iteration)
        out_accumu = 0
        avoid_repeat_list = []
        for f in range(5):
            best = {'column':0, 'bit_w':0, 'loss':9999, 'o':None}
            for column in tqdm(range(images.shape[1]), leave=False):
                if column in avoid_repeat_list:
                    continue
                for bit_w in [-1,1]:
                    rand_swing = torch.randint(0, 2, (images.shape[0],)).cuda().float()
                    o = out_accumu + bit_w * images[:,column] + rand_swing
                    o = (o >= 1).float().unsqueeze(1)
                    r = get_similarity_table(o,o)
                    r = get_classfication_score_table(r, labels, accumulate)
                    r = get_loss(r, labels)
                    if r < best['loss']:
                        best['loss'] = r.item()
                        best['column'] = column
                        best['bit_w'] = bit_w
                        best['o']  = o
                saved_data[iteration,f,0] = best['column']
                saved_data[iteration,f,1] = best['bit_w']
            print('%5d %5d   bit_w:%2d  loss:%8.5f'%\
                    (f,best['column'],best['bit_w'],best['loss']))
            out_accumu += images[:,best['column']] * best['bit_w']
            avoid_repeat_list.append(best['column'])

        o = (out_accumu > 0).float().unsqueeze(1)
        similar_table = get_similarity_table(o,o)
        r = get_classfication_score_table(similar_table, labels, accumulate)
        accumulate = accumulate + similar_table 
        show_gather(best['o'],labels)
        show_accuarcate(r, labels)

        test_o = 0
        for i in range(5):
            test_o += images_t[:,saved_data[iteration, i, 0]] * saved_data[iteration,i,1]
        test_o = (test_o > 0).float().unsqueeze(1)
        similar_table_t = get_similarity_table(test_o,test_o)
        r = get_classfication_score_table(similar_table_t, labels_t, accumulate_t)
        accumulate_t = accumulate_t + similar_table_t
        show_accuarcate(r, labels_t, train=False)

        if IF_SAVE:
            np.save(SAVE_NAME,saved_data)

