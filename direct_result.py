import cv2
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from dataloader import DataLoader
from sklearn.metrics import confusion_matrix
from cradle import Cradle
from entropy import get_images_output,get_loss,show_accuarcate



save_npy_name = 'most_similar_0.5_0.005_7500_200_Ltrainset.npy'
save_npy_name2 = 'most_similar_0.5_0.005_7500_200_Ltrainset_2.npy'
save_npy_name3 = 'most_similar_0.5_0.005_7500_200_Ltrainset_3.npy'


output_save_npy_name = 'direct_output_union_10_layer3.npy'

CLASS = 10
W_LEN = 200
CUDA = 1
CRADLE_SIZE = 50
REPRO_SIZE = 20
UNION = 10

weights = np.load(save_npy_name)
weights = torch.from_numpy(weights).cuda()[:W_LEN]
weights2 = np.load(save_npy_name2)
weights2 = torch.from_numpy(weights2).cuda()[:W_LEN]
weights3 = np.load(save_npy_name3)
weights3 = torch.from_numpy(weights3).cuda()[:W_LEN]



dl = DataLoader(True,CUDA)
images,labels = dl.get_all()
dl_test = DataLoader(False,CUDA)
images_t,labels_t = dl_test.get_all()

images = get_images_output(weights,images)
images_t = get_images_output(weights,images_t)
images = get_images_output(weights2,images)
images_t = get_images_output(weights2,images_t)
images = get_images_output(weights3,images)
images_t = get_images_output(weights3,images_t)



images[images == 0] = -1
images_t[images_t == 0] = -1

images = images.repeat(REPRO_SIZE, 1, 1)
images_t = images_t.repeat(REPRO_SIZE, 1, 1)

cradle = Cradle(CRADLE_SIZE, CLASS * W_LEN * UNION, mutation_rate = 0.005,
            fading_rate = 0.99995,cuda=CUDA)


cradle.from_strach()
best_acc = 0
best_acc_t = 0


def get_output(images, brunch_w):
    w = brunch_w.reshape(REPRO_SIZE,W_LEN,CLASS*UNION)
    r = images.bmm(w) 
    r = (r + W_LEN) / 2
    top_k_elements,_ = torch.topk(r, int(CLASS*UNION*0.2), 2)
    throat,_ = top_k_elements.min(2)
    throat = throat.unsqueeze(2)
    r = (r >= throat).float()
    r = r.reshape(REPRO_SIZE, -1, CLASS, UNION)
    r = r.sum(3)
    r = r.exp()
    r = r/(r.sum(2).unsqueeze(2))
    return r 


for i in range(100000):
    brunch_w = cradle.get_w(REPRO_SIZE)
    r = get_output(images, brunch_w)
    r = get_loss(r, labels)
    cradle.pk(brunch_w,r)
    if i % 100 == 0:
        w = cradle.get_best()[1].cpu().numpy()
        np.save(output_save_npy_name,w)

        w = cradle.get_best()[1].unsqueeze(0).repeat(REPRO_SIZE, 1, 1)
        r = get_output(images, w)
        acc = show_accuarcate(r,labels,train = 0)
        if acc > best_acc:
            best_acc = acc
        w = cradle.get_best()[1].unsqueeze(0).repeat(REPRO_SIZE, 1, 1)
        r = get_output(images_t, w)
        acc = show_accuarcate(r,labels_t,train = 0)
        if acc > best_acc_t:
            best_acc_t = acc

        print('loss:%8.4f   train:%8.4f  test:%8.4f'%(cradle.get_best()[0],best_acc,best_acc_t))

       
