import mnist_web
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
import sys
import time

def visual(image):
    a = image.numpy().reshape(28, 28)
    a = (a + 1) / 2
    cv2.imshow('image', cv2.resize(a, (228,228)))
    key = cv2.waitKey(0)
    if key == ord('q'):
        sys.exit(0)

CLASS = 10
IMAGE_R = 28


class MyDataset(Dataset):
    def __init__(self, train = True):
        if train:
            self.images, self.labels, _, _= mnist_web.mnist(path='.')
            self.MARGIN = 3
        else:
            self.MARGIN = 0
            _, _, self.images, self.labels = mnist_web.mnist(path='.')
        self.l = self.labels.shape[0]
        self.images *= 255
        #self.labels = np.sum(self.labels * np.arange(0,10),1).reshape(-1,1)
        self.images[self.images<=128] = -1
        self.images[self.images>128] = 1
        self.images = torch.from_numpy(self.images).cuda()
        self.labels = torch.from_numpy(self.labels).cuda()
        self.images = self.images.reshape(-1, IMAGE_R, IMAGE_R)
        self.images = torch.nn.functional.pad(self.images, \
                (self.MARGIN,self.MARGIN,self.MARGIN,self.MARGIN), "constant", -1)


        self.accumulate = torch.ones((self.l * ((self.MARGIN  * 2 + 1) ** 2), CLASS)).cuda()
        self.tmp_accum  = torch.zeros((self.l * ((self.MARGIN  * 2 + 1) ** 2))).cuda()
        self.apend_data = torch.zeros((self.l * ((self.MARGIN  * 2 + 1) ** 2), 0)).cuda()

    def update_accumulate(self, mask):
        o = (self.tmp_accum > 0).float().unsqueeze(1)
        o = o * 2 - 1 
        self.accumulate += (o.mm(mask) > 0).float()

    def reset_tmp_accum(self):
        self.tmp_accum *= 0

    def update_tmp_accum(self, column, bit_w):
        for i in range(((self.MARGIN  * 2 + 1) ** 2)):
            a = i // (self.MARGIN  * 2 + 1)
            b = i % (self.MARGIN  * 2 + 1)
            images = self.images[:, a:a+IMAGE_R, b:b+IMAGE_R]
            images = images.reshape(self.l, -1)
            start = i * self.l
            end   = i * self.l + self.l
            data = torch.cat([images, self.apend_data[start:end]], 1)
            self.tmp_accum[start:end] += bit_w * data[:, column]

    def get_accuarcate(self):
        labels = self.labels.repeat(((self.MARGIN  * 2 + 1) ** 2), 1)
        a = torch.argmax(labels, 1)
        b = torch.argmax(self.accumulate,1)
        accuarcate = torch.mean((a==b).float())*100
        return accuarcate

    def __len__(self):
        return self.l * ((self.MARGIN* 2 + 1) ** 2)

    def __getitem__(self, idx):
        image_idx = idx % self.images.shape[0]
        crop_idx = idx // self.images.shape[0]
        a = crop_idx // (self.MARGIN  * 2 + 1)
        b = crop_idx % (self.MARGIN  * 2 + 1)
        image = self.images[image_idx][a:a+IMAGE_R, b:b+IMAGE_R]
        image = image.reshape(-1)
        data = torch.cat([image,self.apend_data[idx]])
        return data, self.accumulate[idx], self.tmp_accum[idx]

if __name__ == '__main__':
    BATCH_SIZE = 10000
    dataset = MyDataset(False)
    dataset.update_tmp_accum(378,-1)
    dataset.update_tmp_accum(350,-1)
    dataset.update_tmp_accum(428, 1)
    dataset.update_tmp_accum(654,-1)
    dataset.update_tmp_accum(658,-1)
    mask = torch.FloatTensor([1,-1,1,-1,1,-1,1,1,-1,1]).unsqueeze(0).cuda()
    dataset.update_accumulate(mask)
    print(dataset.get_accuarcate())
    loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE,\
        shuffle = True, num_workers = 0, drop_last =  True) 
    t1 = time.time()
    for i, a in enumerate(loader):
        print(i, a[0].shape, a[1].shape, a[2].shape)
    t2 = time.time()
    print(t2-t1)

