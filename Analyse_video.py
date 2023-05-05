import cv2
import pyautogui
import time
from time import sleep
import numpy as np
import os
from PIL import Image
import  random as rd

video = cv2.VideoCapture(0)

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)


poids1= np.load('hyper_para/poids_1.npy')
poids2= np.load('hyper_para/poids_2.npy')
poids3= np.load('hyper_para/poids_3.npy')
biais1= np.load('hyper_para/biais_1.npy')
biais2= np.load('hyper_para/biais_2.npy')
biais3= np.load('hyper_para/biais_3.npy')


class Layer_Dense:
    def __init__(self, weights,biaises):
        self.weights = weights
        self.biases = biaises

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        

dense1 = Layer_Dense(poids1, biais1)
dense2 = Layer_Dense(poids2, biais2)
dense3 = Layer_Dense(poids3, biais3)
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Sigmoid()



while True:

    ret, frame = video.read()
    if ret == True:
        result.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  ##interrompre l'analyse
            break

    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    path = 'test_rn/image.png'
    cv2.imwrite(path, image)

    img = Image.open("test_rn/image.png")
    area = (20, 40, 620, 520)
    cropped_img = img.crop(area)
    cropped_img.save('test_rn/image_to_predict.png')

    t1_debut = time.process_time()


    X = []

    path = 'test_rn/img.png'
    image_tp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(image_tp, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    X.append(res)
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2]).astype(np.float32) / 255

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    t_fin = time.process_time()

    if activation3.output > 0.5:
        print("c'est du plastique !",'en {} secondes'.format(t_fin-t1_debut))
    else:
        print("ce n'est pas du plastique !",'En {} secondes'.format(t_fin-t1_debut))

    t_fin,t_debut=0,0



##arnaque