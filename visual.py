import cv2
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from dataloader import DataLoader
from sklearn.metrics import confusion_matrix
from entropy import get_images_output,images,labels,images_t,labels_t,\
        get_similarity_table,get_classfication_score_table, get_loss


CLASS = 10

images_t, labels_t = images_t[:1000], labels_t[:1000]

weights = np.load('most_similar_0.5_0.005_7500_200.npy')
weights = torch.from_numpy(weights).cuda()[:120]


accumulate = torch.zeros((labels_t.shape[0],labels.shape[0]),dtype = torch.float32).cuda()
super_simi_table = torch.zeros((weights.shape[0],labels_t.shape[0],labels.shape[0]),\
        dtype = torch.float32)
super_score_table = torch.torch.zeros((weights.shape[0],labels_t.shape[0],CLASS),\
        dtype = torch.float32)
super_ifcorrect_table = torch.torch.zeros((weights.shape[0],labels_t.shape[0]),\
        dtype = torch.float32)
super_loss_table = torch.torch.zeros((weights.shape[0],labels_t.shape[0]),\
        dtype = torch.float32)
super_confusion_table = torch.torch.zeros((weights.shape[0],CLASS,CLASS),\
        dtype = torch.float32)


def get_loss(class_s_table, labels):
    r = class_s_table / class_s_table.sum(2).unsqueeze(2)
    r = -torch.sum(torch.log(r) * labels,2).squeeze(0)
    return r

def get_ifcoorect(class_s_table, labels):
    a = torch.argmax(class_s_table[0], 1)
    b = labels.argmax(1)
    return (a==b).float()

def get_confusion_table(class_s_table,labels):
    _, a = torch.max(class_s_table[0],1)
    _, b = labels.max(1)
    a = a.cpu()
    b = b.cpu()
    r = confusion_matrix(a, b)
    r = torch.from_numpy(r).float()
    return r

ref_o = get_images_output(weights,images)
test_o = get_images_output(weights,images_t)

print('analyse history...')
pbar = tqdm(total=weights.shape[0])
for i in range(weights.shape[0]):
    o1 = test_o[:,i].unsqueeze(1)
    o2 = ref_o[:,i].unsqueeze(1)
    simi_table = get_similarity_table(o1,o2)
    score = get_classfication_score_table(simi_table, labels, accumulate)
    super_score_table[i] = score.cpu()
    super_loss_table[i] = get_loss(score, labels_t).cpu()
    super_ifcorrect_table[i] = get_ifcoorect(score,labels_t).cpu()
    super_confusion_table[i] = get_confusion_table(score, labels_t)
    accumulate += simi_table.squeeze(0)
    super_simi_table[i] = accumulate.cpu()
    pbar.update(1)
pbar.close()


print(super_loss_table[100])
print(super_score_table[100][:30])
print(super_ifcorrect_table[100][:30])
print(super_confusion_table[100])
