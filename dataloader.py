import mnist_web
import numpy as np
import random
import torch

class DataLoader():
    def __init__(self, train = True, cuda = False):
        if train:
            self.images, self.labels, _, _= mnist_web.mnist(path='.')
        else:
            _, _, self.images, self.labels = mnist_web.mnist(path='.')
        self.pretrain_w = torch.from_numpy(np.load('./maxsume_leaf.npy')).type(torch.int32)
        self.images *= 255
        self.images = self.images.astype('int32')
        self.labels = np.sum(self.labels * np.arange(0,10),1).reshape(-1,1)
        self.labels = self.labels.astype('int32')
        self.images[self.images<=128] = -1
        self.images[self.images>128] = 1
        self.images = torch.from_numpy(self.images)
        self.labels = torch.from_numpy(self.labels)
        if cuda:
            self.images = self.images.cuda().float()
            self.labels = self.labels.cuda()
            self.pretrain_w = self.pretrain_w.cuda().float()
        self.images = torch.mm(self.images, self.pretrain_w)
        self.images[self.images<=0] = -1
        self.images[self.images>0] = 1
                
    def get_all(self):
        return self.images,self.labels

