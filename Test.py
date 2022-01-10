import os  # menejemenisasi file dan folder
import cv2
import numpy as np
from sklearn.utils import shuffle
from PIL import Image  # library clean data
# library untuk menampilkan gambar
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# library tensorflow untuk pelatihan model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # library untuk augmentasi gambar
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as k
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

base_dir = 'datasets/'
# Direktori data daun sakit
diseased_dir = os.path.join(base_dir, 'Deseased')
# Direktori data daun sehat
healthy_dir = os.path.join(base_dir, 'Healthy')
# Mengambil semua anama file dalam masing-masing direktori
diseased_fnames = os.listdir(diseased_dir)
healthy_fnames = os.listdir(healthy_dir)

# check file
print('total data diseased: ', len(os.listdir(diseased_dir)))
print('total data healthy: ', len(os.listdir(healthy_dir)))
#
# cols = 4
# rows = 4
#
# pic_index = 0
# #menampilkan gambar file
# img = plt.gcf()
# img.set_size_inches(cols*4, rows*4)
#
# pic_index+=8
#
# #menyimpan ke dalam list nama file yang akan ditampilkan
# #split
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='training',
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='validation',
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# menampilkan summary/keterangan dari model
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001,
                                                    rho=0.9,
                                                    name="RMSprop"),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

file_path = 'models/'
# Menyimpan model dengan kriteria tertentu
checkpoint = ModelCheckpoint(file_path,
                             save_weights_only=True,
                             )

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    callbacks=[checkpoint],
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    verbose=1
)


def plot_graphs(history, string):
    epochs = range(len(history.history[string]))
    plt.plot(epochs, history.history[string], 'r', label=string)
    plt.plot(epochs, history.history['val_' + string], 'b', label=string + '_val')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

score = model.evaluate(validation_generator, batch_size=10)
print(model.metrics_names)
print(score)
