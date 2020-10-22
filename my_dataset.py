import mnist_web
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
import sys
import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def visual(image):
    a = image.numpy().reshape(28, 28).astype('uint8')
    a = (a + 1) * 127
    cv2.imshow('image', cv2.resize(a, (228,228)))
    key = cv2.waitKey(0)
    if key == ord('q'):
        sys.exit(0)

IMAGE_R = 28
DTYPE = torch.int8


class MyDataset(Dataset):
    def __init__(self, train = True, margin = 0, noise_rate = 0):
        if train:
            self.train = True
            self.images, self.labels, _, _= mnist_web.mnist(path='.')
            self.MARGIN = margin
            self.noise_rate = noise_rate
        else:
            self.train = False
            self.MARGIN = 0
            self.noise_rate = 0
            _, _, self.images, self.labels = mnist_web.mnist(path='.')
        self.labels = np.sum(self.labels * np.arange(0,10),1).reshape(-1,1)
        self.len = self.labels.shape[0]
        self.images *= 255
        self.images[self.images<=128] = -1
        self.images[self.images>128] = 1
        self.images = torch.from_numpy(self.images).type(DTYPE)
        self.images_raw= self.images.clone()
        self.labels = torch.from_numpy(self.labels).type(DTYPE)
        self.images = self.images.reshape(-1, IMAGE_R, IMAGE_R)
        self.images = torch.nn.functional.pad(self.images, \
                (self.MARGIN,self.MARGIN,self.MARGIN,self.MARGIN), "constant", -1)

    def get_all(self):
        return self.images_raw.cuda().float(), self.labels.cuda().type(torch.long).squeeze(-1)

    def __len__(self):
        return 100 * self.len

    def __getitem__(self, idx):
        idx = idx % self.len
        a = random.randint(0, self.MARGIN*2)
        b = random.randint(0, self.MARGIN*2)
        image = self.images[idx][a:a+IMAGE_R, b:b+IMAGE_R]
        image = image.reshape(-1)
        noise_mask = torch.FloatTensor(image.shape[0]).\
                uniform_() > (self.noise_rate)
        noise_mask = noise_mask.type(DTYPE)
        noise_mask = noise_mask * 2 -1
        image = image * noise_mask
        label = self.labels[idx]
        return image, label

class DataFeeder():
    def __init__(self,dataset, batch_size, num_workers):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size,\
            shuffle = True, num_workers= num_workers, drop_last =  True) 
        self.dl_iter = iter(self.dataloader)

    def feed(self):
        try:
            data = next(self.dl_iter)
        except StopIteration:
            self.dl_iter = iter(self.dataloader)
            data  = next(self.dl_iter)
        images, labels = data
        images = images.cuda().float()
        labels = labels.cuda().type(torch.long).squeeze(-1)
        return images, labels

if __name__ == '__main__':
    BATCH_SIZE = 10000
    dataset = MyDataset(True, 3, 0.05)
    feeder = DataFeeder(dataset, BATCH_SIZE, 0)
    for i in range(10):
        t1 = time.time() * 1000
        a = feeder.feed()
        t2 = time.time() * 1000
        print(t2-t1)

