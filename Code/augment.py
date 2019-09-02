''' Script for data augmentation

Marko Rasetina
August 1, 2019
'''

import numpy
import matplotlib.image as mpimg
import os
import cv2


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160


def load_image(data_dir, camera_type):
    return mpimg.imread(os.path.join(data_dir, camera_type.strip()))


def choose_image(data_dir, center, left, right, steering_angle):
    choice = numpy.random.choice(3)

    adjust = 0.12
    if choice == 0:
        return load_image(data_dir, left),  (steering_angle + adjust)
    elif choice == 1:
        return load_image(data_dir, right), (steering_angle - adjust)
    else:
        return load_image(data_dir, center), steering_angle


def random_shadow(image):
    global IMAGE_WIDTH
    global IMAGE_HEIGHT

    x1, y1 = IMAGE_WIDTH * numpy.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * numpy.random.rand(), IMAGE_HEIGHT
    x_m, y_m = numpy.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    mask = numpy.zeros_like(image[:, :, 2])
    mask[(y_m - y1) * (x2 - x1) - (y2 - y1) * (x_m - x1) > 0] = 1

    cond = mask == numpy.random.randint(2)
    s_ratio = 0.35

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2][cond] = hsv[:, :, 2][cond] * s_ratio

    image_shadow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return image_shadow


def augment_data(data_dir, center, left, right, steering):
    image, steering_angle = choose_image(data_dir, center, left, right,
                                         steering)
    image = random_shadow(image)
    return image, steering_angle
