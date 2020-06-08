import torch
import mnist_web
import numpy as np
import random
import sys

class Cradle():
    def __init__(self, cradle_size, total_params, mutation_rate = 0.005, \
                                    fading_rate = 0.9999,cuda=False):
        self.cradle_size = cradle_size
        self.parents_w = None
        self.fading_rate = fading_rate
        self.rank_loss = None
        self.mutation_rate = mutation_rate
        self.total_params = total_params
        self.cuda = cuda 
        self.from_strach()


    def from_strach(self):
        self.best_w = None
        self.best_loss = 9999
        cradle_size = self.cradle_size
        self.parents_w = torch.randint(0, 2, (cradle_size,self.total_params),dtype = torch.int32)
        self.parents_w *= 2
        self.parents_w -= 1 
        self.rank_loss = torch.zeros((cradle_size),dtype = torch.float) + 9999
        if self.cuda:
            self.parents_w = self.parents_w.cuda()
            self.rank_loss = self.rank_loss.cuda()

    
    def get_w(self,bunch_size = 1):
        A = torch.zeros((bunch_size,self.total_params),dtype=torch.int32)
        B = torch.zeros((bunch_size,self.total_params),dtype=torch.int32)
        outputs = torch.randint(0, 2, (bunch_size,self.total_params),dtype = torch.int32)
        outputs = (outputs * 2) - 1
        mutation_mask = torch.randint(0, int(1.0 / self.mutation_rate),\
                (bunch_size,self.total_params),dtype = torch.int32)
        mutation_mask[mutation_mask > 0] = 1
        mutation_mask[mutation_mask == 0] = -1
        if self.cuda:
            A = A.cuda()
            B = B.cuda()
            outputs = outputs.cuda()
            mutation_mask = mutation_mask.cuda()
        for i in range(bunch_size):
            a,b = 0,0
            while(a == b):
                a, b = random.randint(0, self.cradle_size - 1),\
                        random.randint(0, self.cradle_size - 1)
                A[i] = self.parents_w[a]
                B[i] = self.parents_w[b]
                self.rank_loss /= self.fading_rate
        C = A + B
        outputs[C > 0] = 1
        outputs[C < 0] = -1
        outputs *= mutation_mask
        if self.cuda:
            outputs = outputs.float()
            outputs
        return outputs

    def pk(self,bunch_w,bunch_loss):
        for i in range(bunch_w.shape[0]):
            w = bunch_w[i]
            loss = bunch_loss[i]
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_w = w
            worst_pos = torch.argmax(self.rank_loss)
            worst_loss = (self.rank_loss)[worst_pos]
            if loss < worst_loss:
                self.rank_loss[worst_pos] = loss
                self.parents_w[worst_pos] = w

    def set_fading_rate(self,rate):
        self.fading_rate = rate
    
    def get_best(self):
        return self.best_loss,self.best_w


    def get_average_value(self):
        bar = torch.mean(self.rank_loss)
        return bar
    
    def get_parents(self):
        return self.parents_w


#a = Cradle(50,36)
#x = np.array([1,-1,1,-1,-1,1],dtype="int32")
#x = torch.from_numpy(x).reshape(1,-1)
#y = np.array([1,1,-1,1,-1,-1],dtype="int32")
#y = torch.from_numpy(y).reshape(1,-1)
#bunch_size = 4
#
#for i in range(10):
#    b = a.get_w(bunch_size)
#    bunch_loss = torch.zeros(bunch_size)  
#    for j in range(bunch_size):
#        w = b[j]
#        w = w.reshape(6,6)
#        y_predict = torch.mm(x,w)
#        y_predict[y_predict > 0] = 1
#        y_predict[y_predict <= 0] = -1
#        bunch_loss[j] = torch.sum(torch.ne(y,y_predict))
#    a.pk(b,bunch_loss)
#    print(a.get_best_value())
#
