import numpy as np
from PIL import Image, ImageOps
import math

def draw_line(image,x0,y0,x1,y1,color):
    xchange = False
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > x1 - x0) :
            derror -= 2.0 * (x1 - x0) * 1.0
            y += y_update

img_mat = np.zeros((2000,2000,3), dtype = np.uint8) #создеём матрицу
img_mat[...,0] = 255 #делаем красиво фон
img_mat[...,1] = 200
img_mat[...,2] = 177



f = open("model_1.obj")
list = []
paper = []
for s in f:
    splitted = s.split()
    if(splitted[0] == 'v'):
        list.append([float(x) for x in splitted[1:]])
    if (splitted[0] == 'f'):
        paper.append([int(x.split('/')[0]) for x in splitted[1:]])
for vertex in list:
    img_mat[int(10000*vertex[1])+1000,int(10000*vertex[0])+1000] = (141,255,224)
for face in paper: #face - номера вершин i-того треугольника
    x0 = int(10000*list[face[0]-1][0])+1000
    y0 = int(10000*list[face[0]-1][1])+1000
    x1 = int(10000*list[face[1]-1][0])+1000
    y1 = int(10000*list[face[1]-1][1])+1000
    x2 = int(10000*list[face[2]-1][0])+1000
    y2 = int(10000*list[face[2]-1][1])+1000

    draw_line(img_mat,x0,y0,x1,y1,(141,255,224))
    draw_line(img_mat, x0, y0, x2, y2, (141, 255, 224))
    draw_line(img_mat, x2, y2, x1, y1, (141, 255, 224))


img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img3.png')