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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CLASS = 10

dl = DataLoader(True,cuda = 1)
images,labels = dl.get_all()
dl_test = DataLoader(False,cuda = 1)
images_t,labels_t = dl_test.get_all()

def shift_images(images, shift_step):
    a = shift_step
    img = images.reshape(-1, 28, 28)
    img[:, :28-a, :28-a] = img[:, a:,a:].clone()
    img[:, 28-a:, 28-a:] = -1 
    img = img.reshape(-1, 784)
    return img

def add_noise_to_image(images, count):
    img = images.clone()
    mask = torch.cuda.FloatTensor(img.shape).uniform_() > (count / 784.0)
    mask = mask.float()
    mask = (mask * 2) - 1
    img = img * mask
    return img

def visual(before, after):
    a = before.cpu().numpy().reshape(-1, 28, 28)
    b = after.cpu().numpy().reshape(-1, 28, 28)
    a = (a + 1) / 2
    b = (b + 1) / 2
    for i in range(100000):
        cv2.imshow('before', cv2.resize(a[i], (112,112)))
        cv2.imshow('after', cv2.resize(b[i], (112,112)))
        key = cv2.waitKey(0)
        if key == ord('q'):
            sys.exit(0)

before = images.clone()
#images = add_noise_to_image(images, 10)
#images_t = add_noise_to_image(images_t, 10)
images = shift_images(images,1)
images_t = shift_images(images_t,1)
#visual(before, images)

accumulate = torch.ones((labels.shape[0],CLASS),dtype = torch.float32).cuda()
accumulate_t = torch.ones((labels_t.shape[0],CLASS),dtype = torch.float32).cuda()

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


def load_accumulate(inputs, data, mask):
    aide = inputs[:,:data.shape[0]].clone()
    for i in range(data.shape[0]):
        out_sum = 0
        for j in range(5):
            out_sum += inputs[:,data[i, j, 0]] * data[i,j,1]
        result = (out_sum > 0).float()
        result = result * 2 - 1
        aide[:,i] = result
    counter =  mask.abs().sum(0)
    r = aide.mm(mask) 
    r = (r + counter) / 2
    r = r + 1
    return r

LOAD_LEN = 400
data = np.load('five_direct_l3_data.npy')[:LOAD_LEN]
mask = torch.from_numpy((np.load('five_direct_l3_mask.npy')[:LOAD_LEN])).cuda().float()

accumulate = load_accumulate(images, data, mask)
accumulate_t = load_accumulate(images_t, data, mask)



def get_loss(accumulate, labels):
    score_table = torch.exp(accumulate/ 4)
    score_table = score_table / score_table.sum(1).unsqueeze(1)
    loss = -torch.sum(torch.log(score_table) * labels,1)
    loss = loss.mean()
    return loss

def get_accuarcate(r, labels):
    a = torch.argmax(r,1)
    b = labels.argmax(1)
    accuarcate = torch.mean((a==b).float())*100
    return accuarcate


accuarcate = get_accuarcate(accumulate, labels)
train_loss = get_loss(accumulate, labels)
print(train_loss)
print('Train accuarcate:%6.2f%%'%(accuarcate))
accuarcate_t = get_accuarcate(accumulate_t, labels_t)
test_loss = get_loss(accumulate_t, labels_t)
print(test_loss)
print('Test accuarcate:%6.2f%%'%(accuarcate_t))



