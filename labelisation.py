import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2
import matplotlib.pyplot as plt
import random as rd

X = []
Y = []
files = os.listdir('dataset/0')
for image in files:
    path = 'dataset/0/{}'.format(image)
    image_tp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)  # data
    Y.append(0)  # label_entrainement


# data augmentation
def donnee_augmentation_flip(A):
    for i in range(32):
        j = 63 - i
        A[:, [i]], A[:, [j]] = A[:, [j]], A[:, [i]]
    return A


x = rd.randint(100, 1000)
for i in range(x):
    y = rd.randint(0, len(X))
    X.append(donnee_augmentation_flip(X[y]))
    Y.append(0)
files2 = os.listdir('dataset/1')
for image in files2:
    path2 = 'dataset/1/{}'.format(image)
    image_tp2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp2, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)
    Y.append(1)  # label


X, y = np.array(X), np.array(Y).astype('uint8')



np.save('hyper_para/list_image.npy',X)
np.save('hyper_para/list_label.npy',y)