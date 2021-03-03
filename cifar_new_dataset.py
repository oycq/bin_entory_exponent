import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2


SHUFFLE=False
BATCH_SIZE = 100
NUMBER_WORKERS = 14


class Tobits(object):
    def __init__(self):
        pass
        
    def __call__(self, tensor):
        x = tensor * 255
        x = x.numpy().astype(np.uint8)
        x = np.unpackbits(x, axis=0)
        x = torch.from_numpy(x).float()
        return x
    
    def __repr__(self):
        return self.__class__.__name__+'()'


transform = transforms.Compose(
    [
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.2, hue=0.05),
    torchvision.transforms.RandomAffine(degrees=18, translate=(0.1,0.1), scale=(0.9,1.1), shear=None, resample=0, fillcolor=0),
    transforms.ToTensor(),
    Tobits(),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=SHUFFLE, num_workers=NUMBER_WORKERS)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=SHUFFLE, num_workers=NUMBER_WORKERS)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    def imshow(img):
        img = img.permute(1,2,0)
        img = (img*255).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(200,200),interpolation=cv2.INTER_NEAREST)
        cv2.imshow('a',img)
        key = cv2.waitKey(0)
        if key == 'q':
            sys.exit(0)

    for i, data in enumerate(trainloader, 0):
        if i % 100 == 0:
    #        imshow(data[0][0])
            print(data[0].shape)
            print(data[0].type())
            print(data[1].type())



