import cv2
import pyautogui
import time
from time import sleep
import numpy as np
from PIL import Image
import os
import tensorflow as tf

video = cv2.VideoCapture(0)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
model = tf.keras.models.load_model('saved_model/my_model')

while True:
    #lecture flux video
    ret, frame = video.read()
    if ret == True:
        result.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'): ##interrompre l'analyse
            break
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    path = 'test_rn/image_to_predict.png'
    cv2.imwrite(path, image)
    img = Image.open("test_rn/image_to_predict.png")
    area = (20, 40, 620, 550)
    cropped_img = img.crop(area)
    cropped_img.save('test_rn/image.png')
    image_to_predict = cv2.imread('test_rn/image.png', cv2.IMREAD_COLOR)
    img_to_predict = np.expand_dims(cv2.resize(image_to_predict, (64, 64)), axis=0)
    resultat = model.predict(img_to_predict)
    if resultat - 0.5 > 0:
        print("c'est du plastique !")
    else:
        print("ce n'est pas un déchet!")