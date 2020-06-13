import mnist_web
import select
import numpy as np
import random
import sys
import cv2
from dataloader import DataLoader
from cradle import Cradle
import torch

IF_WANDB = 0
if IF_WANDB:
    import wandb
    wandb.init()


dl = DataLoader(train = True)
images, labels = dl.get_all()
format_images = ((images + 1)*127).type(torch.uint8).numpy()
format_images = format_images[:,:,3:-3,3:-3]
w = torch.from_numpy(np.load('./cnn_0.npy')).type(torch.float32)
w = w.permute(1,0)
kernal_n = int(w.shape[1]**0.5)
w = w.reshape(-1,1,kernal_n,kernal_n)

def analyse_images(inputs):
    outputs = torch.nn.functional.conv2d(inputs, w, bias=None,\
            stride=1, padding=0, dilation=1, groups=1)
    outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],-1)
    outputs = (outputs - outputs.mean(2).unsqueeze(-1)) / outputs.std(2).unsqueeze(-1)
    mask = outputs > 1.4
    outputs[mask] = 1
    outputs[~mask] = 0
    outputs_n = int(outputs.shape[2]**0.5)
    outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],outputs_n,outputs_n)
    outputs = outputs.type(torch.uint8).numpy()*255
    for i in range(100):
        for j in range(outputs.shape[1]):
            w_img = ((w[j,0] + 1) * 127).type(torch.uint8).numpy()
            w_img = cv2.resize(w_img,(220,220),interpolation=cv2.INTER_NEAREST)
            input_img = format_images[i,0]
            input_img = cv2.resize(input_img,(220,220),interpolation=cv2.INTER_NEAREST)
            output_img = outputs[i,j]
            output_img = cv2.resize(output_img,(220,220),interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('./img_show/%d%d.jpg'%(10000+j,10000+i),\
                    np.concatenate((input_img,output_img,w_img),1))
#    mask = outputs > 1.4
#    outputs[mask] = 1
#    outputs[~mask] = 0
#    outputs = outputs.sum(2)
#    outputs = (outputs - outputs.mean(0)) / outputs.std(0)
#    mask = outputs > 0
#    outputs[mask] = 1
#    outputs[~mask] = 0

analyse_images(images[:500])
