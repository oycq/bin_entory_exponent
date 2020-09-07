import mnist_web
import select
import numpy as np
import random
import sys
from dataloader import DataLoader
from cradle import Cradle
import torch
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
INPUT_SIZE = 784
CUDA = 1

dl = DataLoader(True,CUDA)
images,labels = dl.get_all()
dl_t = DataLoader(False,CUDA)
images_t,labels_t = dl_t.get_all()

for i in range(30):
    code_l =  44
    w = np.load('./exp1_4.npy')[:code_l,:]
    w = torch.from_numpy(w).type(torch.float32).cuda().t()
    a = images.mm(w)
    a[a>0] = 1
    a[a<=0] = -1
    b = images_t.mm(w).t()
    b[b>0] = 1
    b[b<=0] = -1
    r = a.mm(b)
    r /= 2
    r += code_l / 2
    throat = code_l - i
    a = r > throat
    a = a.float()
    r *= a
    del a 
    r = r ** 1.4
    r = r.mm(labels_t)
    a = r.argmax(1)
    b = labels.argmax(1)
    c = (a == b).float()
    print(i,code_l,c.mean())
