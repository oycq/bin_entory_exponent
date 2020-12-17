import mnist_web
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import cv2
import sys
import time
import cPickle
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class MyDataset(Dataset):
    def __init__(self, train = True, margin = 0, noise_rate = 0):
        self.train = train
        data_dict_list = []
        if train:
            data_dict_list.append(self.unpickle('./cifar_data/data_batch_1'))
            data_dict_list.append(self.unpickle('./cifar_data/data_batch_2'))
            data_dict_list.append(self.unpickle('./cifar_data/data_batch_3'))
            data_dict_list.append(self.unpickle('./cifar_data/data_batch_4'))
            data_dict_list.append(self.unpickle('./cifar_data/data_batch_5'))
        else:
            data_dict_list.append(self.unpickle('./cifar_data/test_batch'))
        self.images_list = []
        self.labels_list = []
        for data_dict in data_dict_list:
            images = data_dict['data']
            images = images.reshape(images.shape[0],3,-1).transpose((0,2,1))
            images = np.unpackbits(images, axis=-1).transpose((0,2,1)).reshape(-1,24,32,32)
            images = torch.from_numpy(images).float()
            labels = torch.zeros(images.shape[0], 10)
            l = data_dict['labels']
            for i in range(len(l)):
                labels[i,l[i]] = 1
            self.images_list.append(images)
            self.labels_list.append(labels)
        self.images = torch.cat(self.images_list, 0)
        self.labels = torch.cat(self.labels_list, 0)
        self.len = self.labels.shape[0]

    def unpickle(self, file_name):
        with open(file_name, 'rb') as fo:
            data_dict = cPickle.load(fo)
        return data_dict


    def get_all(self):
        return self.images.cuda(), self.labels.cuda()

    def __len__(self):
        return 100 * self.len

    def __getitem__(self, idx):
        idx = idx % self.len
        image = self.images[idx]
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
        labels = labels.cuda().float().squeeze(-1)
        return images, labels

if __name__ == '__main__':
    BATCH_SIZE = 1000
    dataset = MyDataset(True, 3, 0.05)
    feeder = DataFeeder(dataset, BATCH_SIZE, 0)
    for i in range(10):
        t1 = time.time() * 1000
        images, lables = feeder.feed()
        t2 = time.time() * 1000
        print(t2-t1)

