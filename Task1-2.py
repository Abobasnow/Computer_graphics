import numpy as np
from PIL import Image
import math

"""
#Версия для лохов
def draw_line(image,x0,y0,x1,y1,color):
    step = 1.0/1000
    for t in np.arange(0,1,step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x, 1] = color
"""
"""
#Версия для продвинутых лохов
def draw_line(image,x0,y0,x1,y1,color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color
"""
"""
#версия не для лохов
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
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
"""
"""
#почти не лохи
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
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 0.5):
            derror -= 1.0
            y += y_update
"""
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
    dy = 2.0*(x1-x0)*abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > 2.0 * (x1 - x0) * 0.5):
            derror -= 2.0 * (x1 - x0) * 1.0
            y += y_update


img_mat = np.zeros((200,200,3), dtype = np.uint8) #создеём матрицу
img_mat[0:600,0:800,0] = 212 #делаем красиво фон
img_mat[0:600,0:800,1] = 107
img_mat[0:600,0:800,2] = 230
for i in range(13):
	x0 = 100
	y0 = 100
	x1 = int(100+95* math.cos((i*2*3.14)/13))
	y1 = int(100+95* math.sin((i*2*3.14)/13))
	draw_line(img_mat,x0,y0,x1,y1,(141,255,224)) #рисуем красиво линии

img = Image.fromarray(img_mat, mode='RGB')
img.save('img.png')
