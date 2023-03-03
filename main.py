import numpy as np
from PIL import Image
import math
import openmesh as om

# # - работа с цветом
# # - синняя картинка
# image = np.zeros((200, 200, 3), dtype=np.uint8)
#
# for i in range(200):
#     for j in range(200):
#         image[i, j, 2] = 255
#         # 0 - красный
#         # 1 - зеленый
#         # 2 - синий
#
# image = Image.fromarray(image)
# image.show()
# image.save("image.jpg")
#
# # - белая
# image = np.zeros((200, 200), dtype=np.uint8)
#
# for i in range(200):
#     for j in range(200):
#         image[i, j] = 255
#
# image = Image.fromarray(image)
# image.show()
# image.save("white_image.jpg")
#
# # - черная
# image = np.zeros((200, 200, 3), dtype=np.uint8)
#
# for i in range(200):
#     for j in range(200):
#         image[i, j] = 0
#
# image = Image.fromarray(image)
# image.show()
# image.save("black_image.jpg")
#
# # - градиент
# image = np.zeros((200, 200, 3), dtype=np.uint8)
#
# for i in range(200):
#     for j in range(200):
#         image[i, j, 0] = i % 256
#         image[i, j, 2] = j
#
# image = Image.fromarray(image)
# image.show()
# image.save("gradient_image.jpg")

# # - отрисовка прямых линий
# # - 1
# image = np.zeros((200,200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = 100 + 95*math.cos((2*math.pi*a)/13),100+95*math.sin((2*math.pi*a)/13)
#     for t_i in range(100):
#         t = 0.01*t_i
#         x = round(int (x0 * t + x1 * (1.0 - t)))
#         y = round(int (y0 * t + y1 * (1.0 - t)))
#         image[x,y] = 255
#
# image = Image.fromarray(image, 'L')
# image.show()
# image.save("lines1.jpg")
#
# # - 2
# image = np.zeros((200,200), dtype=np.uint8)
# for a in range(13):
#     x0, y0 = 100, 100
#     x1, y1 = int(100 + 95*math.cos((2*math.pi*a)/13)),int(100+95*math.sin((2*math.pi*a)/13))
#     for x in range(x0,x1):
#       t = (x-x0)/(x1-x0)
#       y = int(y1 * t + y0 * (1.0 - t))
#       image[y,x] = 255
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

# - модель
mesh = om.read_trimesh('fox.obj', vertex_normal=True)
point_array = mesh.points()
#print(point_array)

image_matrix = np.zeros((1000, 1000), dtype=np.uint8)
for a in range(len(point_array)):
    x0, y0, z0 = point_array[a][0], point_array[a][1], point_array[a][2]
    image_matrix[-int(y0 * 6) + 700, int(z0 * 6) + 570] = 255

image = Image.fromarray(image_matrix, 'L')
# image.show()
# image.save("animal.jpg")

f = open('fox.obj', 'r')
line2 = []
line3 = []
line5 = []
for line in f:
    if line.startswith('f'):
        line2 = line.split(' ')
        line3 = line2[1].split('/') + line2[2].split('/') + line2[3].split('/')
        line4 = line3[0], line3[3], line3[6]
        line5.append(line4)

print(line5)

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


for f in range(len(line5)):
    X0, Y0, Z0 = point_array[int(line5[f][0]) - 1][0], point_array[int(line5[f][0]) - 1][1], \
        point_array[int(line5[f][0]) - 1][2]
    X2, Y2, Z2 = point_array[int(line5[f][2]) - 1][0], point_array[int(line5[f][2]) - 1][1], \
        point_array[int(line5[f][2]) - 1][2]
    X1, Y1, Z1 = point_array[int(line5[f][1]) - 1][0], point_array[int(line5[f][1]) - 1][1], \
        point_array[int(line5[f][1]) - 1][2]
    # print(X0, Y0, X1, Y1)
    x0, y0 = -int(Y0 * 5 + 500), int(Z0 * 5 + 500)
    x2, y2 = -int(Y2 * 5 + 500), int(Z2 * 5 + 500)
    x1, y1 = -int(Y1 * 5 + 500), int(Z1 * 5 + 500)

    foo(x0, x1, y0, y1, image)
    foo(x0, x2, y0, y2, image)
    foo(x1, x2, y1, y2, image)

image = Image.fromarray(image, 'L')
image.show()
image.save("final_animal.jpg")
