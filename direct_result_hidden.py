import cv2
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from dataloader import DataLoader
from sklearn.metrics import confusion_matrix
from cradle import Cradle
from entropy import get_images_output,get_loss,show_accuarcate



save_npy_name = 'most_similar_0.5_0.005_7500_200_Ltrainset.npy'
save_npy_name2 = 'most_similar_0.5_0.005_7500_200_Ltrainset_2.npy'

CLASS = 10
W_LEN = 200
HIDDEN = 50
CUDA = 1
CRADLE_SIZE = 50
REPRO_SIZE = 20 

weights = np.load(save_npy_name)
weights = torch.from_numpy(weights).cuda()[:W_LEN]
weights2 = np.load(save_npy_name2)
weights2 = torch.from_numpy(weights2).cuda()[:W_LEN]


dl = DataLoader(True,CUDA)
images,labels = dl.get_all()
dl_test = DataLoader(False,CUDA)
images_t,labels_t = dl_test.get_all()

images = get_images_output(weights,images)
images_t = get_images_output(weights,images_t)
images = get_images_output(weights2,images)
images_t = get_images_output(weights2,images_t)


images[images == 0] = -1
images_t[images_t == 0] = -1

images = images.repeat(REPRO_SIZE, 1, 1)
images_t = images_t.repeat(REPRO_SIZE, 1, 1)

cradle = Cradle(CRADLE_SIZE, CLASS * HIDDEN + HIDDEN * W_LEN, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)


def w_hidden_output(images, brunch_w):
    w1 = brunch_w[:,:W_LEN * HIDDEN]
    w2 = brunch_w[:,W_LEN* HIDDEN:]
    w1 = w1.reshape(REPRO_SIZE, W_LEN, HIDDEN)
    w2 = w2.reshape(REPRO_SIZE, HIDDEN, CLASS)
    r = images.bmm(w1) 
    r[r>0] = 1
    r[r<=0] = -1
    r = r.bmm(w2)
    r = (r + HIDDEN) / 2
    r = r / (r.sum(2).unsqueeze(2))
    return r



cradle.from_strach()
best_acc = 0
best_acc_t = 0
for i in range(100000):
    brunch_w = cradle.get_w(REPRO_SIZE)
    r = w_hidden_output(images,brunch_w)
    r = get_loss(r, labels)
    cradle.pk(brunch_w,r)
    if i % 100 == 0:
        w = cradle.get_best()[0]
        r = w_hidden_output(images,brunch_w)
        acc = show_accuarcate(r,labels,train = 0)

        w = cradle.get_best()[0]
        r = w_hidden_output(images_t,brunch_w)
        acc_t = show_accuarcate(r,labels_t,train = 0)
        if acc > best_acc:
            best_acc = acc
        if acc_t > best_acc_t:
            best_acc_t = acc_t
        print('loss:%8.4f   train:%8.4f  test:%8.4f'%(cradle.get_best()[0],best_acc,best_acc_t))

       
