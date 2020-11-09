import cv2
import numpy as np
import sys

data1 = np.load('new_sophisticate_layer_2000_0.050_data.npy').astype('int')
data2 = np.load('new_sophisticate_layer_2000_0.051_data.npy').astype('int')
data3 = np.load('new_sophisticate_layer_2000_0.052_data.npy').astype('int')
data = np.concatenate((data1,data2,data3))
colors = [[0,0,1],[0,0,1],[1,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[0,0,0.5],[0,0.5,0]]

point_id = 0
def draw_points(i, data, image):
    global point_id
    for f in range(5):
        column = data[i,f,0]
        if column < 784:
            x = column // 28 
            y = column % 28 
            image[x,y] = colors[data[i,f,1]+1]
            image[x,y] *= (1 - point_id / 200.0)
            point_id += 1
        else:
            draw_points(column-784, data, image)

for i in range(2000):
    i = i + 2000
    image = np.ones((28,28,3))#.astype('uint8') * 255
    draw_points(i, data, image)
    point_id = 0
    cv2.imshow('image', cv2.resize(image,(560,560), interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

