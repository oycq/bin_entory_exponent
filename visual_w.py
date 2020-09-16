import cv2
import numpy as np
import sys

npy_name = sys.argv[1]


while(1):
    try:
        w = np.load(npy_name)
        w[w== 1] = 0
        w[w==-1] = 1
        R = int(w.shape[1]**0.5)
        w = w.reshape(w.shape[0],R,R)

        curtain = np.zeros((1500,1500)) + 0.5
        for i in range(10):
            for j in range(10):
                k = i * 10 + j
                img = cv2.resize(w[k],(100,100),interpolation=cv2.INTER_NEAREST)
                curtain[i*150:i*150+100,j*150:j*150+100] = img
        cv2.imshow('curtain',curtain)
        key = cv2.waitKey(1000)
        if key == ord('q'):
            break
    except:
        pass

#for i in range(w.shape[0]):




