import random
import numpy as np
from PIL import Image, ImageOps
import math

# Параметры изображения
IMG_SIZE_H = 2000  # Высота изображения
IMG_SIZE_W = 6000  # Ширина изображения
PROJ_COEFF = 7200
FOCAL_OFFSET = 1000

# Инициализация z-буфера
A = np.full((IMG_SIZE_W, IMG_SIZE_H), np.inf, dtype=np.float32)

def bary(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    l2 = 1.0 - l0 - l1
    return l0, l1, l2

def norma(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    i = (y1 - y2) * (z1 - z0) - (y1 - y0) * (z1 - z2)
    j = (z1 - z2) * (x1 - x0) - (z1 - z0) * (x1 - x2)
    k = (x1 - x2) * (y1 - y0) - (x1 - x0) * (y1 - y2)
    dlina = (i ** 2 + j ** 2 + k ** 2) ** 0.5
    return i / dlina, j / dlina, k / dlina

def draw_triangle(img_mat, x0, y0, x1, y1, x2, y2, z0, z1, z2, k1, k2, k3, u0, u1, u2, v0, v1, v2, texture, WT, HT):
    a = PROJ_COEFF
    u00, v00 = ((a * x0) / z0) + FOCAL_OFFSET, ((a * y0) / z0) + FOCAL_OFFSET
    u11, v11 = ((a * x1) / z1) + FOCAL_OFFSET, ((a * y1) / z1) + FOCAL_OFFSET
    u22, v22 = ((a * x2) / z2) + FOCAL_OFFSET, ((a * y2) / z2) + FOCAL_OFFSET
    xmin = int(min(u00, u11, u22))
    if xmin < 0:
        xmin = 0

    ymin = int(min(v00, v11, v22))
    if ymin < 0:
        ymin = 0

    xmax = int(max(u00, u11, u22))
    if xmax >= IMG_SIZE_W:
        xmax = IMG_SIZE_W - 1

    ymax = int(max(v00, v11, v22))
    if ymax >= IMG_SIZE_H:
        ymax = IMG_SIZE_H - 1

    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            l0, l1, l2 = bary(u00, v00, u11, v11, u22, v22, i, j)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                u = int(WT * (u0 * l0 + u1 * l1 + u2 * l2))
                v = int(HT * (v0 * l0 + v1 * l1 + v2 * l2))
                colour = texture[v, u]
                zv = l0 * z0 + l1 * z1 + l2 * z2
                if A[i][j] > zv:
                    wap = l0 * k1 + l1 * k2 + l2 * k3
                    img_mat[j, i] = colour * (-wap)
                    A[i][j] = zv

def quaternion_multiply(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([a, b, c, d])

def quaternion_conjugate(q):
    a, b, c, d = q
    return np.array([a, -b, -c, -d])

def rotate_vector_by_quaternion(v, q):
    p = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    qp = quaternion_multiply(q, p)
    rotated_p = quaternion_multiply(qp, q_conj)
    return rotated_p[1:]

#                   ЭТО НАЧАЛО ПРОГРАММЫ САШААААААААААААААААААААААААААААААа

# Загрузка текстуры
texture_img = ImageOps.flip(Image.open("bunny-atlas.jpg"))
texture = np.array(texture_img)
WT, HT = texture.shape[1] , texture.shape[0]

# Инициализация матрицы изображения
img_mat = np.zeros((IMG_SIZE_H, IMG_SIZE_W, 3), dtype=np.uint8)
img_mat[..., 0] = 64
img_mat[..., 1] = 130
img_mat[..., 2] = 109

# Парсинг .obj файла
def parse_obj(file_path):
    vertices = []
    tex_coords = []
    faces = []
    tex_indices = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:])))
            elif parts[0] == 'f':
                polygon = []
                for entry in parts[1:]:
                    indices = entry.split('/')
                    v_idx = int(indices[0]) - 1 if indices[0] else -1
                    t_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                    polygon.append((v_idx, t_idx))

                if len(polygon) > 3:
                    for i in range(1, len(polygon) - 1):
                        faces.append([polygon[0][0], polygon[i][0], polygon[i + 1][0]])
                        tex_indices.append([polygon[0][1], polygon[i][1], polygon[i + 1][1]])
                else:
                    faces.append([polygon[0][0], polygon[1][0], polygon[2][0]])
                    tex_indices.append([polygon[0][1], polygon[1][1], polygon[2][1]])
            elif parts[0] == 'vt':
                tex_coords.append(list(map(float, parts[1:])))

    return vertices, tex_coords, faces, tex_indices

def place_model_on_canvas(vertices, tex_coords, faces, tex_indices, img_mat, A, texture, WT, HT, tx, ty, tz, scale, use_quaternion=False):
    # Применение ротации, сдвига и масштаба к вершинам
    if use_quaternion:
        # Определите ось вращения и угол
        axis = np.array([1.0, 1.0, 1.0])  # Пример оси вращения
        axis = axis / np.linalg.norm(axis)  # Нормализация оси
        angle = np.pi / 4  # Угол поворота в радианах

        # Вычисление кватерниона
        q0 = np.cos(angle / 2)
        q1 = axis[0] * np.sin(angle / 2)
        q2 = axis[1] * np.sin(angle / 2)
        q3 = axis[2] * np.sin(angle / 2)
        quaternion = np.array([q0, q1, q2, q3])
    else:
        # Углы Эйлера
        angle_x = -np.pi / 3
        angle_y = np.pi / 2
        angle_z = np.pi / 4

        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)

        m1 = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        m2 = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        m3 = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = np.dot(m3, np.dot(m2, m1))

    transformed_vertices = []
    for i in range(len(vertices)):
        scaled_vertex = np.array(vertices[i]) * scale
        if use_quaternion:
            rotated_vertex = rotate_vector_by_quaternion(scaled_vertex, quaternion)
            transformed_vertices.append(rotated_vertex + np.array([tx, ty, tz]))
        else:
            transformed_vertices.append(np.dot(R, scaled_vertex) + np.array([tx, ty, tz]))

    # нормали
    vn_calc = np.zeros((len(transformed_vertices), 3))

    for face in faces:
        x0 = transformed_vertices[face[0]][0]
        y0 = transformed_vertices[face[0]][1]
        z0 = transformed_vertices[face[0]][2]
        x1 = transformed_vertices[face[1]][0]
        y1 = transformed_vertices[face[1]][1]
        z1 = transformed_vertices[face[1]][2]
        x2 = transformed_vertices[face[2]][0]
        y2 = transformed_vertices[face[2]][1]
        z2 = transformed_vertices[face[2]][2]
        nx, ny, nz = norma(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        vn_calc[face[0]] += np.array([nx, ny, nz])
        vn_calc[face[1]] += np.array([nx, ny, nz])
        vn_calc[face[2]] += np.array([nx, ny, nz])

    for i in range(len(vn_calc)):
        dlina = np.linalg.norm(vn_calc[i])
        vn_calc[i] /= dlina

    # Отрисовка треугольников
    for (face, plank) in zip(faces, tex_indices):
        x0 = transformed_vertices[face[0]][0]
        y0 = transformed_vertices[face[0]][1]
        z0 = transformed_vertices[face[0]][2]
        x1 = transformed_vertices[face[1]][0]
        y1 = transformed_vertices[face[1]][1]
        z1 = transformed_vertices[face[1]][2]
        x2 = transformed_vertices[face[2]][0]
        y2 = transformed_vertices[face[2]][1]
        z2 = transformed_vertices[face[2]][2]
        u0 = tex_coords[plank[0]][0]
        v0 = tex_coords[plank[0]][1]
        u1 = tex_coords[plank[1]][0]
        v1 = tex_coords[plank[1]][1]
        u2 = tex_coords[plank[2]][0]
        v2 = tex_coords[plank[2]][1]
        cx, cy, cz = norma(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cz0 = vn_calc[face[0]][2]
        cz1 = vn_calc[face[1]][2]
        cz2 = vn_calc[face[2]][2]

        if cz < 0:
            draw_triangle(img_mat, x0, y0, x1, y1, x2, y2, z0, z1, z2, cz0, cz1, cz2, u0, u1, u2, v0, v1, v2, texture, WT, HT)


vertices, tex_coords, faces, tex_indices = parse_obj("model_1.obj")
tx, ty, tz = 0, 0.05, 3
scale = 1.8
use_quaternion = True  #  False
place_model_on_canvas(vertices, tex_coords, faces, tex_indices, img_mat, A, texture, WT, HT, tx, ty, tz, scale, use_quaternion)

vertices2, tex_coords2, faces2, tex_indices2 = parse_obj("banjofrog.obj")
texture2_img = ImageOps.flip(Image.open("banjofrog.jpg"))
texture2 = np.array(texture2_img)
WT2, HT2 = texture2.shape[1] , texture2.shape[0]
tx2, ty2, tz2 = 0.5, 0.05, 4
scale2 = 0.1
use_quaternion2 = False
place_model_on_canvas(vertices2, tex_coords2, faces2, tex_indices2, img_mat, A, texture2, WT2, HT2, tx2, ty2, tz2, scale2, use_quaternion2)


# Сохранение изображения
img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img19.png')