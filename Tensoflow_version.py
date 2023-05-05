# Imports needed
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_height = 64
img_width =64
batch_size = 30


ds_train =  tf.keras.preprocessing.image_dataset_from_directory(
  'dataset',
  validation_split=0.2,
    shuffle=True,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

ds_validation =  tf.keras.preprocessing.image_dataset_from_directory(
  'dataset',
  validation_split=0.2,
    shuffle=True,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)



class_names = ds_validation.class_names



model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255), #norlalisation
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

class_names = ds_train.class_names

model.compile(optimizer='adam',loss=tf.losses.BinaryCrossentropy(from_logits=True),metrics=['BinaryAccuracy'],)
logdir="logs"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir, embeddings_data=ds_train)

model.fit(
    ds_train,
  validation_data=ds_validation,
  epochs=2,
  callbacks=[tensorboard_callback]
)

model.save('saved_model/my_model')
