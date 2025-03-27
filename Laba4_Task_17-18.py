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

texture_img = ImageOps.flip(Image.open("bunny-atlas.jpg"))
texture = np.array(texture_img)
WT, HT = texture_img.size

def draw_triangle(img_mat, x0,y0,x1,y1,x2,y2,z0,z1,z2,k1,k2,k3,u0,u1,u2,v0,v1,v2):
    a = 7200
    u00, v00 = ((a*x0)/z0) + 1000, ((a*y0)/z0) + 1000
    u11, v11 = ((a*x1)/z1) + 1000, ((a*y1)/z1) + 1000
    u22, v22 = ((a*x2)/z2) + 1000, ((a*y2)/z2) + 1000
    xmin = int(min(u00,u11,u22))
    if(xmin<0): xmin = 0

    ymin = int(min(v00,v11,v22))
    if(ymin<0): ymin = 0

    xmax = int(max(u00,u11,u22))
    if(xmax>=2000):xmax=2000-1

    ymax = int(max(v00, v11, v22))
    if (ymax >= 2000): ymax = 2000-1

    #colour = (255*abs(kos), 146*abs(kos), 24*abs(kos))

    for i in range(xmin,xmax+1):
        for j in range(ymin, ymax+1):
            l0,l1,l2 = bary(u00,v00,u11,v11,u22,v22,i,j)
            if(l0>=0 and l1>=0 and l2>=0):
                u = int(WT*(u0*l0+u1*l1+u2*l2))
                v = int(HT*(v0*l0+v1*l1+v2*l2))
                colour = texture[v,u]
                zv = l0*z0+ l1*z1 + l2*z2
                if(A[i][j]>zv):
                    bam = l0*k1+l1*k2+l2*k3
                    img_mat[j,i] = colour*(-bam)
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
foil = []
trellis = [] #шпалера
for s in f:
    splitted = s.split()
    if(splitted[0] == 'v'):
        list.append([float(x) for x in splitted[1:]])
    if (splitted[0] == 'f'):
        paper.append([int(x.split('/')[0]) for x in splitted[1:]])
        foil.append([int(x.split('/')[1]) for x in splitted[1:]]) #номера вершин
    if( splitted[0] == 'vt'):
        trellis.append([float(x) for x in splitted[1:]]) #координаты теней

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

vn_calc = np.zeros((len(list), 3))

for rock in paper: #вычисляем нормали и запихиваем в массив нормалей
    x0 = (list[rock[0] - 1][0])
    y0 = (list[rock[0] - 1][1])
    z0 = (list[rock[0] - 1][2])
    x1 = (list[rock[1] - 1][0])
    y1 = (list[rock[1] - 1][1])
    z1 = (list[rock[1] - 1][2])
    x2 = (list[rock[2] - 1][0])
    y2 = (list[rock[2] - 1][1])
    z2 = (list[rock[2] - 1][2])
    nx,ny,nz = norma(x0,y0, z0, x1, y1, z1, x2, y2, z2)
    vn_calc[rock[0] - 1][0] += nx
    vn_calc[rock[0] - 1][1] += ny
    vn_calc[rock[0] - 1][2] += nz
    vn_calc[rock[1] - 1][0] += nx
    vn_calc[rock[1] - 1][1] += ny
    vn_calc[rock[1] - 1][2] += nz
    vn_calc[rock[2] - 1][0] += nx
    vn_calc[rock[2] - 1][1] += ny
    vn_calc[rock[2] - 1][2] += nz

for i in range(len(vn_calc)):
    dlina = (vn_calc[i][0]**2 + vn_calc[i][1]**2 + vn_calc[i][2]**2)**0.5
    vn_calc[i][0]/= dlina
    vn_calc[i][1] /= dlina
    vn_calc[i][2] /= dlina

for (face, plank) in zip(paper, foil): #face - номера вершин i-того треугольника ЭТО КСТАТИ РЕНДЕР ЧТО Б ТЫ ЗНАЛ
    x0 = list[face[0]-1][0]
    y0 = list[face[0]-1][1]
    z0 = list[face[0]-1][2]
    x1 = list[face[1]-1][0]
    y1 = list[face[1]-1][1]
    z1 = list[face[1]-1][2]
    x2 = list[face[2]-1][0]
    y2 = list[face[2]-1][1]
    z2 = list[face[2]-1][2]
    u0 = trellis[plank[0]-1][0]
    v0 = trellis[plank[0]-1][1]
    u1 = trellis[plank[1] - 1][0]
    v1 = trellis[plank[1] - 1][1]
    u2 = trellis[plank[2] - 1][0]
    v2 = trellis[plank[2] - 1][1]
    cx, cy, cz = norma(x0,y0, z0, x1, y1, z1, x2, y2, z2) #координаты нормали к полигону
    cz0 = vn_calc[face[0] - 1][2]
    cz1 = vn_calc[face[1] - 1][2]
    cz2 = vn_calc[face[2] - 1][2]

    if(cz<0): draw_triangle(img_mat,x0,y0,x1,y1,x2,y2,z0,z1,z2,cz0,cz1,cz2,u0,u1,u2,v0,v1,v2)




    #draw_triangle(img_mat, x0, y0, x2, y2)
    #draw_triangle(img_mat, x2, y2, x1, y1)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img17.png')


