import mnist_web
import select
import numpy as np
import random
import sys

random.seed(0)
np.random.seed(4)
#w1 = np.load('w1.npy')
project_name = 'w1_bunch'

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

class Cradle():
    def __init__(self,k, mutation_rate = 0.005, fading_rate = 0.9999):
        self.k = k
        self.w_id = -1
        self.total_params = 0
        self.w_start_end = []
        self.current_w = None
        self.parents_w = None
        self.fading_rate = fading_rate
        self.top_values = None
        self.mutation_rate = mutation_rate
    
    
    def set_fading_rate(self,rate):
        self.fading_rate = rate
    
    def register(self,size):
        self.w_id += 1
        self.w_start_end.append([self.total_params,self.total_params + size])
        self.total_params += size
        return self.w_id 
    
    def get_w(self,w_id):
        a, b = self.w_start_end[w_id]
        w = self.current_w[a:b]
        return w
    
    def get_parents(self):
        return self.parents_w

    def reproduce(self):
        a,b = 0,0
        while(a == b):
            a, b = random.randint(0, self.k - 1), random.randint(0, self.k - 1)
        A, B = self.parents_w[a], self.parents_w[b]
        add = A + B 
        self.current_w = np.random.choice(a=[-1, 1], size=(self.total_params), p=[0.5, 0.5]) 
        self.current_w[add > 0] = 1
        self.current_w[add < 0] = -1
        mutation_mask = np.random.choice(a=[-1, 1], size=(self.total_params), \
                                    p=[self.mutation_rate, 1 - self.mutation_rate])
        self.current_w *= mutation_mask
        self.top_values /= self.fading_rate
        return

    def pk(self, value):
        max_pos = np.argmax(self.top_values)
        max_grade = (self.top_values)[max_pos]
        if value < max_grade:
            self.top_values[max_pos] = value
            self.parents_w[max_pos] = self.current_w

    def from_strach(self):
        a = self.total_params
        k = self.k
        self.parents_w = np.random.choice(a=[-1, 1], size=(k,a), p=[0.5, 0.5])  
        self.top_values = np.zeros((k)) + 9999
    
    def save(self,path):
        np.save('%s.npy'%project_name,self.parents_w)
        pass
    
    def load(self,path):
        k = self.k
        self.parents_w = np.load('%s/res.npy'%path)
        self.top_values = np.zeros((k)) + 9999

def my_loss(outputs,labels,debug=0):
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
            if debug:
                a = np.max(np.sum(group[1],0)) 
                print(i,a,items_count,np.sum(group[1],0))
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
    return loss, correct_rate, labels_ptable

def caculate_output(inputs,w):
    ww = np.concatenate((prior_w, w), axis=1)
    output = np.dot(inputs,ww)
    output[output > 0] = 1
    output[output <= 0] = -1
    return output

w_width = 784 
w_n = 1 
BATCH_SIZE = 60000
CRADLE_N = 50

cradle =  Cradle(CRADLE_N, mutation_rate = 0.005, fading_rate = 0.99995)
dl = DataLoader(BATCH_SIZE, train=True)
w_id = cradle.register(w_width * w_n)
cradle.from_strach()
prior_w = np.zeros((w_width,0),dtype=int)

j = -1
nnn = 1
while(1):
    if nnn == 12:
        break
    j += 1
    if j == 10:
        cradle.set_fading_rate(0.99999)
    if j == 30:
        cradle.set_fading_rate(0.999995)
    if j == 50:
        cradle.set_fading_rate(0.999999)
    if j == 100:
        cradle.set_fading_rate(0.9999995)
    for i in range(500):
        inputs, labels = dl.get_batch()
        #inputs, labels = dl.get_all()
        cradle.reproduce()
        w = cradle.get_w(w_id)
        w = w.reshape(w_width,w_n)
        output = caculate_output(inputs,w)
        loss,correct_rate,labels_ptable = my_loss(output, labels)
        cradle.pk(loss)
#    log_file = open('%s.log'%project_name,'a')
#    log_file.write("%d   %10.4f  %10.4f %10.4f\n"\
#          %(j,average_top_loss,np.min(cradle.top_values),correct_rate))
#    log_file.close()
    average_top_loss = np.average(cradle.top_values)
    print("%d   %10.4f  %10.4f %10.4f"\
          %(j,average_top_loss,np.min(cradle.top_values),correct_rate))

    if j == 100:
        print('get it')
        inputs, labels = dl.get_all()
        parents_w = cradle.get_parents()
        best_loss = 9999
        best_w = None
        for i in range(CRADLE_N):
            w = parents_w[i]
            w = w.reshape(w_width,w_n)
            output = caculate_output(inputs,w)
            loss,correct_rate,labels_ptable = my_loss(output, labels)
            if loss < best_loss:
                best_loss = loss
                best_w = w
        output = caculate_output(inputs,best_w)
        best_loss,best_correct_rate,best_ptable = my_loss(output, labels,1)
        prior_w = np.concatenate((prior_w, best_w), axis=1)
        print('best loss: %8.3f correct_rate: %8.3f'%(best_loss,best_correct_rate))
        cradle =  Cradle(CRADLE_N, mutation_rate = 0.005, fading_rate = 0.99995)
        w_id = cradle.register(w_width * w_n)
        cradle.from_strach()
        j = -1
        nnn += 1
        np.save('entropy.npy',prior_w)


 
#    if select.select([sys.stdin,],[],[],0.0)[0]:
#        bar = input()
#        if bar == 1:
#            print('get it')
#            inputs, labels = dl.get_all()
#            parents_w = cradle.get_parents()
#            best_loss = 9999
#            best_w = None
#            for i in range(CRADLE_N):
#                w = parents_w[i]
#                w = w.reshape(w_width,w_n)
#                output = caculate_output(inputs,w)
#                loss,correct_rate,labels_ptable = my_loss(output, labels)
#                if loss < best_loss:
#                    best_loss = loss
#                    best_w = w
#            output = caculate_output(inputs,best_w)
#            best_loss,best_correct_rate,best_ptable = my_loss(output, labels,1)
#            prior_w = np.concatenate((prior_w, best_w), axis=1)
#            print('best loss: %8.3f correct_rate: %8.3f'%(best_loss,best_correct_rate))
#            cradle =  Cradle(CRADLE_N, mutation_rate = 0.005, fading_rate = 0.99995)
#            w_id = cradle.register(w_width * w_n)
#            cradle.from_strach()
#            j = -1
#
