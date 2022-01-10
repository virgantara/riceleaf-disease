import os #menejemenisasi file dan folder
import cv2
import numpy as np
from sklearn.utils import shuffle
from PIL import Image #library clean data
#library untuk menampilkan gambar
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#library tensorflow untuk pelatihan model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #library untuk augmentasi gambar
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

#Direktori data
base_dir = 'dataset'
#Direktori data daun sakit
blast_dir = os.path.join(base_dir, 'Blast')
#Direktori data daun sehat
brownspot_dir = os.path.join(base_dir, 'Brown_Spot')
#Direktori data daun sehat
hispa_dir = os.path.join(base_dir, 'Hispa')
#Mengambil semua anama file dalam masing-masing direktori
balst_fnames = os.listdir(blast_dir )
brownspot_fnames = os.listdir(brownspot_dir)
hispa_fnames = os.listdir(hispa_dir)

#check file
print('total data Blast:',len(os.listdir(blast_dir)))
print('total data Brownspot:',len(os.listdir(brownspot_dir)))
print('total data Hispa:',len(os.listdir(hispa_dir)))

cols = 4
rows = 3

pic_index = 0
#menampilkan gambar file
img = plt.gcf()
img.set_size_inches(cols*4, rows*3)

pic_index+=4

#menyimpan ke dalam list nama file yang akan ditampilkan
show_blast_img = [os.path.join(blast_dir, fname)
                      for fname in balst_fnames[pic_index-4:pic_index]
                    ]

show_brownspot_img = [os.path.join(brownspot_dir, fname)
                      for fname in brownspot_fnames[pic_index-4:pic_index]
                    ]
show_hispa_img = [os.path.join(hispa_dir, fname)
                      for fname in hispa_fnames[pic_index-4:pic_index]
                    ]
# for i, img_path in enumerate(show_blast_img+show_brownspot_img+show_hispa_img):
#     sp = plt.subplot(rows, cols, i + 1)
#     sp.axis('off')
#
#     image = mpimg.imread(img_path)
#     plt.imshow(image)
#
# plt.show()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='wrap',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224,224),
    shuffle=True,
    batch_size = 32,
    subset='training',
    class_mode = 'categorical'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224,224),
    batch_size = 32,
    subset='validation',
    class_mode = 'categorical'
)

print(train_generator.class_indices)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation= 'softmax')
])

#menampilkan summary/keterangan dari model
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
file_path = 'models/model_checkpoint-{epoch:02d}-{accuracy:.02f}.h5'
#Menyimpan model dengan kriteria tertentu
checkpoint = ModelCheckpoint(file_path,
                             monitor='val_loss',
                             save_best_only = True,
                             mode = 'min'
                            )

history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs = 50,
    callbacks=[checkpoint],
    validation_data = validation_generator,
    validation_steps =  len(validation_generator),
    verbose = 2
)

score = model.evaluate(validation_generator, batch_size=32)
print(model.metrics_names, score)

def plot_graphs(history, string):
    epochs = range(len(history.history[string]))
    plt.plot(epochs, history.history[string],'r', label=string)
    plt.plot(epochs, history.history['val_'+string],'b',label=string+'_val')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()

plot_graphs(history,"accuracy")
plot_graphs(history,"loss")