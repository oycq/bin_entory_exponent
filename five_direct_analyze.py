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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
CLASS = 10

dl = DataLoader(True,cuda = 1)
images,labels = dl.get_all()
dl_test = DataLoader(False,cuda = 1)
images_t,labels_t = dl_test.get_all()

accumulate = torch.ones((labels.shape[0],CLASS),dtype = torch.float32).cuda()
accumulate_t = torch.ones((labels_t.shape[0],CLASS),dtype = torch.float32).cuda()


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

LOAD_LEN = 1 
data = np.load('five_direct_v1_data.npy')[:LOAD_LEN]
mask = torch.from_numpy((np.load('five_direct_v1_mask.npy')[:LOAD_LEN])).cuda().float()
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


class Enhacne(nn.Module):
    def __init__(self):
        super(Enhacne, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for i,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                   nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = Enhacne().cuda()
optimizer = optim.Adam(model.parameters())
for i in range(10000):
    r = model(accumulate/50)
    loss = get_loss(r, labels)
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    acc = get_accuarcate(r, labels)
    if i % 100 == 0:
        r = model(accumulate_t/50)
        loss_t = get_loss(r, labels_t)
        acc_t = get_accuarcate(r, labels_t)
        print(i)
        print('train %8.4f %9.4f'%(loss,acc))
        print('test  %8.4f %9.4f'%(loss_t,acc_t))
        print('')

    

    





