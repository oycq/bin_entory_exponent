import mnist_web
import numpy as np
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
        
    def get_batch(self):
        #shaffle to batch
        batch_mask = np.random.choice(self.n, self.batch_size, replace=False)
        return self.images[batch_mask],self.labels[batch_mask]
                
    def get_all(self):
        return self.images,self.labels


def generate_table(outputs,labels):
    n = outputs.shape[1]
    labels_len = labels.shape[1]
    groups = [(outputs,labels)]
    labels_ptable = np.zeros((2 ** n,labels_len))
    labels_ptable += 1.0 / labels_len
    entropy = 0
    correct_count = 0
    for i in range(n):
        new_groups = []
        for group in groups:
            o = group[0]
            l  = group[1]
            mask0 = o[:,i] == -1
            mask1 = o[:,i] ==  1
            new_groups.append((o[mask0],l[mask0]))
            new_groups.append((o[mask1],l[mask1]))
        groups = new_groups
    for i,group in enumerate(groups):
        items_count = group[0].shape[0]
        if items_count == 0:
            continue
        else:
            correct_count += np.max(np.sum(group[1],0))
            labels_p = np.sum(group[1],0) * 1.0 / items_count 
            labels_ptable[i] = labels_p
            e = labels_p.copy()
            e = e[e>0]
            e = - e * np.log(e) #/ np.log(2)
            e = np.sum(e) 
            e *= items_count
            entropy += e
    entropy /= outputs.shape[0]
    correct_rate = correct_count * 1.0 / outputs.shape[0]
    loss = entropy
    labels_table = np.argmax(labels_ptable, 1)
    return labels_table


def caculate_correct_rate(outputs,labels,labels_tables):
    n = outputs.shape[1]
    outputs[outputs == -1] = 0
    table_pointer = outputs[:,-1] * 0
    for i in range(n):
        a = n - i - 1
        table_pointer += 2 ** a * outputs[:,i]
    predict = labels_tables[table_pointer]
    labels = labels.astype(int)
    for i in range(labels.shape[1]):
        labels[:,i] *= i
    labels = np.sum(labels, 1)
    correct_rate = np.average(labels == predict)
    return correct_rate

for i in range(16): # max 16
    w1_n  = i + 1
    print('when w1_n = %5d'%w1_n)
    dl_train = DataLoader(train = True) 
    w1 = np.load('./entropy_16.npy')
    w1 = w1[:,:w1_n]
    inputs,labels = dl_train.get_all()
    outputs = np.dot(inputs, w1)
    outputs[outputs > 0] = 1
    outputs[outputs <= 0] = -1
    labels_table = generate_table(outputs,labels)
    train_correct = caculate_correct_rate(outputs,labels,labels_table)
    print('train_correct_rate : %8.2f%%'%(train_correct*100))

    dl_test = DataLoader(train = False) 
    inputs,labels = dl_test.get_all()
    inputs,labels = dl_test.get_all()
    outputs = np.dot(inputs, w1)
    outputs[outputs > 0] = 1
    outputs[outputs <= 0] = -1
    test_correct = caculate_correct_rate(outputs,labels,labels_table)
    print('test__correct_rate : %8.2f%%'%(test_correct*100))

