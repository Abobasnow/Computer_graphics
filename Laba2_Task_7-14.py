import random

import numpy as np
from PIL import Image, ImageOps
import math

A = np.full((2000,2000),np.inf,dtype = np.float32)

def bary(x0,y0,x1,y1,x2,y2,x,y):
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0,l1,l2

def draw_triangle(img_mat, x0,y0,x1,y1,x2,y2, kos,z0,z1,z2):
    xmin = int(min(x0,x1,x2))
    if(xmin<0): xmin = 0

    ymin = int(min(y0,y1,y2))
    if(ymin<0): ymin = 0

    xmax = int(max(x0,x1,x2))
    if(xmax>=2000):xmax=2000-1

    ymax = int(max(y0, y1, y2))
    if (ymax >= 2000): ymax = 2000-1

    colour = (255*abs(kos), 0, 0)

    for i in range(xmin,xmax+1):
        for j in range(ymin, ymax+1):
            l0,l1,l2 = bary(x0,y0,x1,y1,x2,y2,i,j)
            if(l0>=0 and l1>=0 and l2>=0):
                zv = l0*z0+ l1*z1 + l2*z2
                if(A[i][j]>zv):
                    img_mat[j,i] = colour
                    A[i][j] = zv

def norma(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    i = (y1-y2)*(z1-z0) - (y1-y0)*(z1-z2)
    j = (z1-z2)*(x1-x0) - (z1-z0)*(x1-x2)
    k = (x1-x2)*(y1-y0) - (x1-x0)*(y1-y2)
    dlina = (i ** 2 + j ** 2 + k ** 2) ** 0.5
    return i/dlina,j/dlina,k/dlina



img_mat = np.zeros((2000,2000,3), dtype = np.uint8) #создеём матрицу
img_mat[...,0] = 209 #делаем красиво фон
img_mat[...,1] = 52
img_mat[...,2] = 99

#draw_triangle(img_mat, 0,0, 200,477, 2000,2000)

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
    x0 = (10000*list[face[0]-1][0])+1000
    y0 = (10000*list[face[0]-1][1])+1000
    z0 = (10000*list[face[0]-1][2])+1000
    x1 = (10000*list[face[1]-1][0])+1000
    y1 = (10000*list[face[1]-1][1])+1000
    z1 = (10000*list[face[1]-1][2])+1000
    x2 = (10000*list[face[2]-1][0])+1000
    y2 = (10000*list[face[2]-1][1])+1000
    z2 = (10000*list[face[2]-1][2])+1000
    nx,ny,nz = norma(x0,y0, z0, x1, y1, z1, x2, y2, z2)
    if(nz<0): draw_triangle(img_mat,x0,y0,x1,y1,x2,y2,nz,z0,z1,z2)


    #draw_triangle(img_mat, x0, y0, x2, y2)
    #draw_triangle(img_mat, x2, y2, x1, y1)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img8.png')


