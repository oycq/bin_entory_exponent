import mnist_web
import numpy as np
import random
import sys
from cradle import Cradle
import torch
import time
from tqdm import tqdm
import my_dataset 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

IF_WANDB = 0
IF_SAVE = 0
LAYER_UNITS = 2000
LAYERS = 3 
CLASS = 10
BATCH_SIZE = 100
NAME = 'neural_400_100'
WORKERS = 15

if IF_WANDB:
    import wandb
    wandb.init(project = 'neural', name = NAME)

#dataset = my_dataset.MyDataset(train = True, margin = 3, noise_rate = 0.05)
dataset_test = my_dataset.MyDataset(train = False)
#data_feeder = my_dataset.DataFeeder(dataset, BATCH_SIZE, num_workers = WORKERS)
images_t,labels_t = dataset_test.get_all()


class Quantized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()


    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input.abs() > 1)] = 0
        return grad_input 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.s1 = nn.Sequential(
            nn.Linear(5, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace = True),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace = True),
            nn.Linear(200, 1),
            #nn.BatchNorm1d(1)
            )

        self.s2 = nn.Sequential(
            nn.Linear(1, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace = True),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace = True),
            nn.Linear(200, 10))

    def forward(self, inputs, quant = 0):
        a = []
        quantized = Quantized.apply
        x = self.s1(inputs)
        #x = nn.Sigmoid()(x*1)
        regu = (x ** 2 - 1).abs().mean()

        if quant:
            x = quantized(x)

        for i in range(1000):
            if x[i].item() not in a:
                a.append(x[i].item())
        x = self.s2(x)
        return x, regu, a

    def direct_forward(self, inputs):
        return self.s2(inputs)

def get_loss(x, labels):
    acc = (x.argmax(-1) == labels.argmax(-1)).float().mean() * 100
    x = x.exp()
    x = x / x.sum(-1).unsqueeze(-1)
    x = -x.log()
    loss = (x * labels).sum(-1)
    loss = loss.mean()
    return loss, acc

net = Net().cuda()
#optimizer = optim.Adam(net.parameters())
#optimizer = optim.Adadelta(net.parameters())
optimizer = optim.Adagrad(net.parameters())
select_features = torch.LongTensor([378,461,401,541,464]).cuda()
        
for i in range(10000):
    #x = net.direct_forward(images_t[:,select_features])
    x, regu, a = net(images_t[:,select_features], 1)

    loss, acc = get_loss(x, labels_t)
    regu *= 0.1
    loss += regu
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(i)
        print('%9.3f %9.3f %9.3f'%(regu, loss-regu, acc))
        a.sort()
        for item in a:
            print('%9.4f'%item)
    if i > 5000:
        x, regu, a = net(images_t[:,select_features], quant = 1)
        loss, acc = get_loss(x, labels_t)
        print(i)
        print('%9.3f %9.3f %9.3f'%(regu, loss, acc))
        a.sort()
        for item in a:
            print('%9.4f'%item)
        print(x.argmax(-1))
        print(labels_t.argmax(-1))

        break




