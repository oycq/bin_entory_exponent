import cv2
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from dataloader import DataLoader
from sklearn.metrics import confusion_matrix
from entropy import get_images_output,images,labels,images_t,labels_t,\
        get_similarity_table,get_classfication_score_table, get_loss,similar_k_rate


CLASS = 10
CURTAIN_W = 3600
CURTAIN_H = 1800
IMAGES_PER_PAGE = 15
W_LEN = 120

images_t, labels_t = images_t[:1000], labels_t[:1000]

weights = np.load('most_similar_0.5_0.005_7500_200.npy')
weights = torch.from_numpy(weights).cuda()[:W_LEN]


accumulate = torch.zeros((labels_t.shape[0],labels.shape[0]),dtype = torch.float32).cuda()
super_simi_table = torch.zeros((weights.shape[0],labels_t.shape[0],labels.shape[0]),\
        dtype = torch.float32)
super_score_table = torch.zeros((weights.shape[0],labels_t.shape[0],CLASS),\
        dtype = torch.float32)
super_ifcorrect_table = torch.zeros((weights.shape[0],labels_t.shape[0]),\
        dtype = torch.float32)
super_loss_table = torch.zeros((weights.shape[0],labels_t.shape[0]),\
        dtype = torch.float32)
super_confusion_table = torch.zeros((weights.shape[0],CLASS,CLASS),\
        dtype = torch.float32)
super_confusion_label = torch.zeros((weights.shape[0],labels_t.shape[0]),\
        dtype = torch.long)



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

def get_confusion_label(class_s_table, labels):
    index = torch.arange(0, labels.shape[0])
    _, a = torch.max(class_s_table[0],1)
    _, b = labels.max(1)
    c = a.cpu() * CLASS + b.cpu() + index * CLASS * CLASS
    return c 


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
    super_confusion_label[i] = get_confusion_label(score, labels_t)
    accumulate += simi_table.squeeze(0)
    super_simi_table[i] = accumulate.cpu()
    pbar.update(1)
pbar.close()

#front-end
image_page = 0
image_ref_page = 0

image_page_i = 0
images_ref_page_i =0

w_i = 0
image_pointer = 0
confusion_pointer = 0
lock_focus = 0
focused_image_id = 0

images_t = ((images_t + 1) * 127).type(torch.uint8)
images_t = images_t.unsqueeze(2).repeat(1,1,3).cpu()
images = ((images + 1) * 127).type(torch.uint8)
images = images.unsqueeze(2).repeat(1,1,3).cpu()
weights = ((weights + 1) * 127).type(torch.uint8)
weights = weights.unsqueeze(2).repeat(1,1,3).cpu()

def draw_selected_images(curtain,selected_images,start_pos_k,page_base,hight_light):
    selected_images = selected_images.numpy()
    img_a = int(selected_images.shape[1]**0.5)
    for i in range(IMAGES_PER_PAGE):
        l_box = int(CURTAIN_W/IMAGES_PER_PAGE)
        l = int(CURTAIN_W/IMAGES_PER_PAGE * 0.9)
        j = page_base * IMAGES_PER_PAGE + i
        if j < selected_images.shape[0]:
            a = l_box * i
            b = l + l_box * i
            c = int(start_pos_k * CURTAIN_H)
            d = c + l
            img = selected_images[j].reshape(img_a,img_a,3)
            curtain[c:d,a:b] = cv2.resize(img,(l,l),interpolation = cv2.INTER_NEAREST)
            if (j == hight_light) and (j != -1):
                image = cv2.rectangle(curtain, (a,c), (b,d), (255,0,0), 3) 



def draw_confusion_images(curtain,start_pos_k):
    global image_pointer,focused_image_id
    select = (super_confusion_label[w_i]%(CLASS*CLASS)) == confusion_pointer
    selected_images = images_t[select]
    if image_pointer >= selected_images.shape[0]:
        image_pointer = selected_images.shape[0] - 1 
    if image_pointer < 0:
        image_pointer = 0
    image_page = image_pointer / IMAGES_PER_PAGE
    draw_selected_images(curtain,selected_images,start_pos_k,image_page,image_pointer)
    if (not lock_focus) and (sum(select)):
        focused_image_id = super_confusion_label[w_i][select][image_pointer] / (CLASS * CLASS)

def draw_ref_images(curtain,start_pos_k):
    score, index = torch.topk(super_simi_table[w_i][focused_image_id],\
            int(images.shape[0]*similar_k_rate))
    selected_images = torch.index_select(images, 0, index)
    draw_selected_images(curtain,selected_images,start_pos_k,image_ref_page,-1)

def draw_w(curtain,start_pos_k):
    w_page = w_i / IMAGES_PER_PAGE
    #selected_images = weights[w_page*IMAGES_PER_PAGE:(1+w_page)*IMAGES_PER_PAGE]
    selected_images = weights
    draw_selected_images(curtain,selected_images,start_pos_k,w_page,w_i)

def draw_string_in_roi(string,roi):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (0,0)
    fontScale = 1
    thickness = 2
    color = (0, 0, 0) 

    lines = string.split('\n')
    lines_n = len(lines)
    delta_h = int(roi.shape[0] * 1.0 / lines_n)
    for i in range(lines_n):
        org = (0,org[1]+delta_h)
        line = lines[i].replace('\n','')
        cv2.putText(roi, line, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    return roi




def draw_confusion_table(curtain,s_hk,s_wk,hk,wk):
    confusion_table = super_confusion_table[w_i]
    string = ''
    for i in range(CLASS):
        line = ''
        for j in range(CLASS):
            if (i * CLASS + j) == confusion_pointer:
                line += '%5d*'%confusion_table[i][j]
            else:
                line += '%6d'%confusion_table[i][j]
        string = string + line + '\n\n'
    a = int(s_hk * CURTAIN_H)
    b = a + int(hk * CURTAIN_H)
    c = int(s_wk * CURTAIN_W)
    d = c + int(wk * CURTAIN_W)
    roi = curtain[a:b,c:d]
    draw_string_in_roi(string,roi)
    return curtain


while(1):
    curtain = np.zeros((CURTAIN_H, CURTAIN_W,3),dtype=np.uint8) + 128
    draw_confusion_images(curtain,0)
    draw_ref_images(curtain,1.0*CURTAIN_W/IMAGES_PER_PAGE/CURTAIN_H)
    draw_w(curtain,1 - 1.0*CURTAIN_W/IMAGES_PER_PAGE/CURTAIN_H)
    curtain = draw_confusion_table(curtain, 0.32,0.7,0.5,0.3)

    cv2.imshow('ui',curtain)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    if key == ord('7'):
        confusion_pointer -= 1
    if key == ord('9'):
        confusion_pointer += 1
    if key == ord('/'):
        confusion_pointer -= CLASS
    if key == ord('8'):
        confusion_pointer += CLASS
    if key == ord('1'):
        w_i -= 1
    if key == ord('3'):
        w_i += 1
    if key == ord('2'):
        w_i += IMAGES_PER_PAGE
    if key == ord('5'):
        w_i -= IMAGES_PER_PAGE

    if key == ord('a'):
        image_pointer -= 1
    if key == ord('d'):
        image_pointer += 1
    if key == ord('w'):
        image_pointer -= IMAGES_PER_PAGE
    if key == ord('s'):
        image_pointer += IMAGES_PER_PAGE
    if key == ord('\\'):
        lock_focus = (lock_focus + 1) % 2

    confusion_pointer %= (CLASS*CLASS)
    if w_i < 0:
        w_i = 0
    if w_i >= weights.shape[0]:
        w_i = weights.shape[0] - 1  
    
