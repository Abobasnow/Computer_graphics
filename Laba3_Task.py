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
    a = 7200
    u0, v0 = ((a*x0)/z0) + 1000, ((a*y0)/z0) + 1000
    u1, v1 = ((a*x1)/z1) + 1000, ((a*y1)/z1) + 1000
    u2, v2 = ((a*x2)/z2) + 1000, ((a*y2)/z2) + 1000
    xmin = int(min(u0,u1,u2))
    if(xmin<0): xmin = 0

    ymin = int(min(v0,v1,v2))
    if(ymin<0): ymin = 0

    xmax = int(max(u0,u1,u2))
    if(xmax>=2000):xmax=2000-1

    ymax = int(max(v0, v1, v2))
    if (ymax >= 2000): ymax = 2000-1

    colour = (255*abs(kos), 146*abs(kos), 24*abs(kos))

    for i in range(xmin,xmax+1):
        for j in range(ymin, ymax+1):
            l0,l1,l2 = bary(u0,v0,u1,v1,u2,v2,i,j)
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
img_mat[...,0] = 190 #делаем красиво фон
img_mat[...,1] = 245
img_mat[...,2] = 116

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

for i in range (len(list)):
    #img_mat[int(10000*vertex[1])+1000,int(10000*vertex[0])+1000] = (141,255,224)
    tx,ty,tz = 0,0.05,0.8
    ccx = -0.1
    ccy = 0.4
    ccz = 0.2
    sx = (1 - ccx ** 2) ** 0.5
    sy = (1 - ccy ** 2) ** 0.5
    sz = (1 - ccz ** 2) ** 0.5
    m1 = np.array([[1, 0, 0], [0, ccx, sx], [0, -sx, ccx]])
    m2 = np.array([[ccy, 0, sy], [0, 1, 0], [-sy, 0, ccy]])
    m3 = np.array([[ccx, sz, 0], [-sy, ccz, 0], [0, 0, 1]])
    R = np.dot(np.dot(m1, m2), m3)
    list[i] = np.dot(R,list[i]) + np.array([tx,ty,tz])

for face in paper: #face - номера вершин i-того треугольника ЭТО КСТАТИ РЕНДЕР ЧТО Б ТЫ ЗНАЛ
    x0 = (list[face[0]-1][0])
    y0 = (list[face[0]-1][1])
    z0 = (list[face[0]-1][2])
    x1 = (list[face[1]-1][0])
    y1 = (list[face[1]-1][1])
    z1 = (list[face[1]-1][2])
    x2 = (list[face[2]-1][0])
    y2 = (list[face[2]-1][1])
    z2 = (list[face[2]-1][2])
    cx,cy,cz = norma(x0,y0, z0, x1, y1, z1, x2, y2, z2) #это косинусы
    if(cz<0): draw_triangle(img_mat,x0,y0,x1,y1,x2,y2,cz,z0,z1,z2)


    #draw_triangle(img_mat, x0, y0, x2, y2)
    #draw_triangle(img_mat, x2, y2, x1, y1)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img12.png')


