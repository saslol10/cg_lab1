import numpy as np
from PIL import Image
import math
import random


# # 1. Работа с изображениями.
# # - черная
# image = np.zeros((200, 200, 3), dtype=np.uint8)
# for i in range(200):
#     for j in range(200):
#         image[i, j] = 0
#
# image = Image.fromarray(image)
# image.show()
# image.save("black_image.jpg")
#
# # - белая
# image = np.zeros((200, 200), dtype=np.uint8)
# for i in range(200):
#     for j in range(200):
#         image[i, j] = 255
#
# image = Image.fromarray(image)
# image.show()
# image.save("white_image.jpg")
#
# # - красная
# image = np.zeros((200, 200, 3), dtype=np.uint8)
# for i in range(200):
#     for j in range(200):
#         image[i, j, 0] = 255
#         # 0 - красный
#         # 1 - зеленый
#         # 2 - синий
#
# image = Image.fromarray(image)
# image.show()
# image.save("red_image.jpg")
#
# # - градиент
# image = np.zeros((200, 200, 3), dtype=np.uint8)
#
# for i in range(200):
#     for j in range(200):
#         image[i, j] = [((i / 2) + (j / 2)) % 256, 0, 0]
#
# image = Image.fromarray(image)
# image.show()
# image.save("gradient_image.jpg")
#
#
# # 2. Работа с изображениями (ООП)
# class Colour:
#     colour_array = [0, 0, 0]
#
#     def __init__(self, colour_array):
#         self.colour_array = colour_array
#
#
# class Picture:
#     h = 512
#     w = 512
#
#     picture_array = np.zeros((h, w, 3), dtype=np.uint8)
#     default_colour = Colour([128, 128, 128])  # цвет фона
#     picture_colour = Colour([0, 0, 0])  # цвет изображения
#
#     def __init__(self, h, w, col: Colour):
#         self.h = h
#         self.w = w
#         self.picture_array = np.zeros((h, w, 3), dtype=np.uint8)
#         self.picture_colour = col
#         self.clear()
#         self.z_matrix = np.zeros((h, w))  # Матрица для проверки и дальйнешего удаления невидимых поверхностей
#
#     # создание изображения
#     def create_from_array(self, array):
#         self.picture_array = array
#
#     # задание цвета конкретного пикселя изображения
#     def set_pixel(self, x, y, color: Colour):
#         # if self.w > x > 0 and self.h > y > 0:
#         self.picture_array[int(-y), int(-x)] = color.colour_array
#
#     # вывод изображения
#     def show_picture(self):
#         img = Image.fromarray(self.picture_array, 'RGB')
#         img.show()
#
#     # удаление изображения
#     def clear(self):
#         self.picture_array[0:self.h, 0:self.w] = self.default_colour.colour_array
#
#
# # 3. Отрисовка прямых линий
# # - 1
# image = np.zeros((200, 200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = 100 + 95 * math.cos((2 * math.pi * a) / 13), 100 + 95 * math.sin((2 * math.pi * a) / 13)
#     for t_i in range(100):
#         t = 0.01 * t_i
#         x = round(int(x0 * t + x1 * (1.0 - t)))
#         y = round(int(y0 * t + y1 * (1.0 - t)))
#         image[x, y] = 255
#
# image = Image.fromarray(image, 'L')
# image.show()
# image.save("lines1.jpg")
#
# # - 2
# image = np.zeros((200, 200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = int(100 + 95 * math.cos((2 * math.pi * a) / 13)), int(100 + 95 * math.sin((2 * math.pi * a) / 13))
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = int(y1 * t + y0 * (1.0 - t))
#         image[y, x] = 255
#
# image = Image.fromarray(image, 'L')
# image.show()
# image.save("lines2.jpg")
#
# # - 3
# image = np.zeros((200, 200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = int(100 + 95 * math.cos((2 * math.pi * a) / 13)), int(100 + 95 * math.sin((2 * math.pi * a) / 13))
#     steep = False
#
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         steep = True
#
#     if (x0 > x1):
#         x0, x1, = x1, x0
#         y0, y1 = y1, y0
#
#     for x in range(x0, x1):
#         if (steep):
#             t = (x - x0) / (x1 - x0)
#             y = int(y0 * (1.0 - t) + y1 * t)
#             image[y, x] = 255
#             image[x, y] = 255
# image = Image.fromarray(image, 'L')
# image.show()
# image.save("lines3.jpg")
#
# # - 4 (Брезенхема)
# image = np.zeros((200, 200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = int(100 + 95 * math.cos((2 * math.pi * a) / 13)), int(100 + 95 * math.sin((2 * math.pi * a) / 13))
#     steep = False
#
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         steep = True
#
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     dx = x1 - x0
#     dy = y1 - y0
#     derror = abs(dy / dx)
#     error = 0
#     y = y0
#     x = x0
#     for x in range(x0, x1):
#         if (steep):
#             image[y, x] = 255
#         else:
#             image[x, y] = 255
#
#         error += derror
#         if (error > .5):
#             if (y1 > y0):
#                 y += 1
#             else:
#                 y -= 1
#
#             error -= 1.
# image = Image.fromarray(image, 'L')
# image.show()
# image.save("lines4.jpg")

# 4. - 7. Работа с трёхмерной моделью
class Model_Object:
    def __init__(self, points=[], edges=[]):
        self.points = points
        self.edges = edges

    def read_file(self, filename):

        with open(filename) as file:
            lines = file.readlines()

            for line in lines:
                if line.startswith('v '):  # 4. Работа с трёхмерной моделью (вершины)
                    x, y, z = line[2:].split(' ')
                    self.points.append((float(x), float(y), float(z)))
                elif line.startswith('f '):  # 6. Работа с трёхмерной моделью (полигоны)
                    edge = line[2:].split(' ')

                    if len(edge) == 4:
                        v1, v2, v3, _ = edge
                    else:
                        v1, v2, v3 = edge
                    num_of_point1, num_of_point2, num_of_point3 = map(int, (
                        v1.split('/')[0], v2.split('/')[0], v3.split('/')[0]))
                    self.edges.append((num_of_point1, num_of_point2, num_of_point3))


model = Model_Object()
model.read_file('model_1.obj')
# 5. Отрисовка вершин трёхмерной модели
image_matrix = np.zeros((1000, 1000), dtype=np.uint8)
for v in range(len(model.points)):
    x0, y0, z0 = model.points[v][0], model.points[v][1], model.points[v][2]
    image_matrix[-int(y0 * 5) + 500, int(z0 * 5) + 500] = 255

image = Image.fromarray(image_matrix, 'L')
image.show()
image.save("model_1-points.jpg")

image = np.zeros((1000, 1000), dtype=np.uint8)


def foo(x0, x1, y0, y1, image):
    steep = False

    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    # print(x0,x1,y0,y1)
    # print(f)
    # print(line5[f])
    if (x0 == x1):
        return
    derror = abs(dy / dx)
    error = 0
    y = y0
    x = x0
    for x in range(x0, x1):
        if (steep):
            image[y, x] = 255
        else:
            image[x, y] = 255

        error += derror
        if (error > .5):
            if (y1 > y0):
                y += 1
            else:
                y -= 1

            error -= 1.


# 7. Отрисовка рёбер трёхмерной модели
for f in range(len(model.edges)):
    X0, Y0, Z0 = model.points[int(model.edges[f][0]) - 1][0], model.points[int(model.edges[f][0]) - 1][1], \
        model.points[int(model.edges[f][0]) - 1][2]
    X2, Y2, Z2 = model.points[int(model.edges[f][2]) - 1][0], model.points[int(model.edges[f][2]) - 1][1], \
        model.points[int(model.edges[f][2]) - 1][2]
    X1, Y1, Z1 = model.points[int(model.edges[f][1]) - 1][0], model.points[int(model.edges[f][1]) - 1][1], \
        model.points[int(model.edges[f][1]) - 1][2]
    # print(X0, Y0, X1, Y1)
    x0, y0 = -int(Y0 * 5 + 500), int(Z0 * 5 + 500)
    x2, y2 = -int(Y2 * 5 + 500), int(Z2 * 5 + 500)
    x1, y1 = -int(Y1 * 5 + 500), int(Z1 * 5 + 500)

    foo(x0, x1, y0, y1, image)
    foo(x0, x2, y0, y2, image)
    foo(x1, x2, y1, y2, image)


image = Image.fromarray(image, 'L')
image.show()
image.save("model_1-edges.jpg")


# 8. Барицентрические координаты
def baricentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return (lambda0, lambda1, lambda2)


# 9. Отрисовка треугольников
def draw_triangle(img, x0, x1, x2, y0, y1, y2):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    for i in range(int(xmin), int(xmax) + 1):
        for j in range(int(ymin), int(ymax) + 1):
            lambda0, lambda1, lambda2 = baricentric(i, j, x0, y0, x1, y1, x2, y2)
            if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
                img[i, j] = [r, g, b]


# 10. Тестирование функции
x0 = model.points[0][0]
x1 = model.points[1][0]
x2 = model.points[2][0]

y0 = model.points[0][1]
y1 = model.points[1][1]
y2 = model.points[2][1]

z0 = model.points[0][2]
z1 = model.points[1][2]
z2 = model.points[2][2]

img = np.zeros((1000, 1000, 3), dtype=np.uint8)
draw_triangle(img, 5 * x0 + 500, 5 * x1 + 500, 5 * x2 + 500, 5 * y0 + 500, 5 * y1 + 500, 5 * y2 + 500)

image = Image.fromarray(img)
image.show()
image.save("triangle-10.jpg")

# 11
img = np.zeros((1000, 1000, 3), dtype=np.uint8)

for edge in model.edges:
    x0, y0, z0 = model.points[edge[0] - 1]
    x1, y1, z1 = model.points[edge[1] - 1]
    x2, y2, z2 = model.points[edge[2] - 1]

    x0, y0, z0 = (-y0 * 5 + 500), (z0 * 5 + 500), (x0 * 5 + 500)
    x1, y1, z1 = (-y1 * 5 + 500), (z1 * 5 + 500), (x1 * 5 + 500)
    x2, y2, z2 = (-y2 * 5 + 500), (z2 * 5 + 500), (x2 * 5 + 500)

    draw_triangle(img, x0, x1, x2, y0, y1, y2)

image = Image.fromarray(img)
image.show()
image.save("model_1-11.jpg")


# 12. - 15.
def draw_triangle(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, z_buff, light=[0, 0, 1]):
    if x0 == x1 and x1 == x2 or y0 == y1 and y1 == y2:
        return

    normal = np.cross([x1 - x0, y1 - y0, z1 - z0], [x1 - x2, y1 - y2, z1 - z2])

    if np.dot(normal, light) >= 0:
        return

    color = (-255 * np.dot(normal, light) / np.linalg.norm(normal) / np.linalg.norm(light), 0, 0)

    x0, y0, z0 = project_coordinates(x0, y0, z0, u0, v0, a_x, a_y, t_z)
    x1, y1, z1 = project_coordinates(x1, y1, z1, u0, v0, a_x, a_y, t_z)
    x2, y2, z2 = project_coordinates(x2, y2, z2, u0, v0, a_x, a_y, t_z)

    xmin = 0 if min(x0, x1, x2) < 0 else min(x0, x1, x2)
    ymin = 0 if min(y0, y1, y2) < 0 else min(y0, y1, y2)
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1, y2)

    for i in range(int(xmin), int(xmax) + 1):
        for j in range(int(ymin), int(ymax) + 1):
            lambda0, lambda1, lambda2 = baricentric(i, j, x0, y0, x1, y1, x2, y2)

            if lambda0 > 0 and lambda1 > 0 and lambda2 > 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                # if i >= len(z_buff) or j >= len(z_buff):
                #     continue
                if z_buff[i][j] < z:
                    continue
                img[i][j] = color
                z_buff[i][j] = z


H = 1000
W = 1000


def project_coordinates(x, y, z, u0, v0, a_x, a_y, t_z):
    x, y, z = np.array([[a_x, 0, u0],
                        [0, a_y, v0],
                        [0, 0, 1]]) @ (np.array([x, y, z]) + np.array([0.005, -0.045, t_z]))
    return x / 10000, y / 10000, z/10000


img = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf)

u0 = 500
v0 = 500

a_x = 100
a_y = 100

t_z = 0.1

for edge in model.edges:
    x0, y0, z0 = model.points[edge[0] - 1]
    x1, y1, z1 = model.points[edge[1] - 1]
    x2, y2, z2 = model.points[edge[2] - 1]

    x0, y0, z0 = np.array([x0, y0, z0])
    x1, y1, z1 = np.array([x1, y1, z1])
    x2, y2, z2 = np.array([x2, y2, z2])

    draw_triangle(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, z_buff=z_buffer)

image = Image.fromarray(img)
image.show()
image.save("model_1-16.jpg")

# # 17. Поворот модели:
# H = 1000
# W = 1000
# img = np.zeros((H, W, 3), dtype=np.uint8)
# z_buffer = np.full((H, W), np.inf)
#
# for edge in model.edges:
#     x0, y0, z0 = model.points[edge[0] - 1]
#     x1, y1, z1 = model.points[edge[1] - 1]
#     x2, y2, z2 = model.points[edge[2] - 1]
#
#     x0, y0, z0 = (-y0 * 5 + 500), (-x0 * 5 + 500), (z0 * 5 + 500)
#     x1, y1, z1 = (-y1 * 5 + 500), (-x1 * 5 + 500), (z1 * 5 + 500)
#     x2, y2, z2 = (-y2 * 5 + 500), (-x2 * 5 + 500), (z2 * 5 + 500)
#
#     draw_triangleNext(img, x0, y0, z0, x1, y1, z1, x2, y2, z2, z_buff=z_buffer)
#
# image = Image.fromarray(img)
# image.show()
# image.save("fox-17.jpg")
