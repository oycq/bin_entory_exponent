import mnist_web
import numpy as np
import random
import sys

class Cradle():
    def __init__(self,cradle_size, mutation_rate = 0.005, fading_rate = 0.9999):
        self.cradle_size = cradle_size
        self.w_id = -1
        self.total_params = 0
        self.w_start_end = []
        self.current_w = None
        self.parents_w = None
        self.fading_rate = fading_rate
        self.rank_values = None
        self.mutation_rate = mutation_rate
    
    
    def set_fading_rate(self,rate):
        self.fading_rate = rate
    
    def register(self,w_len):
        self.w_id += 1
        self.w_start_end.append([self.total_params,self.total_params + w_len])
        self.total_params += w_len 
        return self.w_id 
    
    def get_w(self,w_id):
        a, b = self.w_start_end[w_id]
        w = self.current_w[a:b]
        return w
    
    def get_best_value(self):
        bar = np.min(self.rank_values)
        return bar

    def get_average_value(self):
        bar = np.mean(self.rank_values)
        return bar
    
    def get_parents(self):
        return self.parents_w

    def reproduce(self,bunch_size = 1):
        a,b = 0,0
        while(a == b):
            a, b = random.randint(0, self.cradle_size - 1),\
                    random.randint(0, self.cradle_size - 1)
        A, B = self.parents_w[a], self.parents_w[b]
        add = A + B 
        self.current_w = np.random.choice(a=[-1, 1], \
                size=(self.total_params), p=[0.5, 0.5]) 
        self.current_w[add > 0] = 1
        self.current_w[add < 0] = -1
        mutation_mask = np.random.choice(a=[-1, 1], size=(self.total_params),\
                p=[self.mutation_rate, 1 - self.mutation_rate])
        self.current_w *= mutation_mask
        self.rank_values /= self.fading_rate
        return

    def bunch_pk(self,wv_list):
        for item in wv_list:
            w = item[0]
            w = w.reshape(w.size)
            value = item[1]
            max_pos = np.argmax(self.rank_values)
            max_grade = (self.rank_values)[max_pos]
            if value < max_grade:
                self.rank_values[max_pos] = value
                self.parents_w[max_pos] = w

    def from_strach(self):
        a = self.total_params
        cradle_size = self.cradle_size
        self.parents_w = np.random.choice(a=[-1, 1], \
                size=(cradle_size,a), p=[0.5, 0.5])  
        self.rank_values = np.zeros((cradle_size)) + 9999
    
    def save(self,file_name):
        np.save('%s.npy'%file_name,self.parents_w)
        pass
    
    def load(self,file_name):
        cradle_size = self.cradle_size
        self.parents_w = np.load('%s/res.npy'%file_name)
        self.rank_values = np.zeros((cradle_size)) + 9999

