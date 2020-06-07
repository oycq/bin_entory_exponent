import mnist_web
import numpy as np
import random
import sys

class DataLoader():
    def __init__(self,batch_size = 1000, train = True):
        self.batch_size = batch_size
        self.classes = 10
        if train:
            self.images, self.labels, _, _= mnist_web.mnist(path='.')
        else:
            _, _, self.images, self.labels = mnist_web.mnist(path='.')
        self.n = self.images.shape[0]
        self.images *= 255
        self.images = self.images.astype(int)
        self.images[self.images<=128] = -1
        self.images[self.images>128] = 1
        #dot w1
        #self.images = np.dot(self.images,w1)
    
    def get_batch(self):
        #shaffle to batch
        batch_mask = np.random.choice(self.n, self.batch_size, replace=False)
        return self.images[batch_mask],self.labels[batch_mask]
                
    def get_all(self):
        return self.images,self.labels


