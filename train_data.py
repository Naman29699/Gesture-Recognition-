import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

train_dir = r"C:\Users\naman\Desktop\vlc\train"
test_dir = r"C:\Users\naman\Desktop\vlc\test"

train_up_dir = r"C:\Users\naman\Desktop\vlc\train\up"
train_down_dir = r"C:\Users\naman\Desktop\vlc\train\down"
train_left_dir = r"C:\Users\naman\Desktop\vlc\train\left"
train_nothing_dir = r"C:/Users/naman/Desktop/vlc/train/nothing"
train_right_dir = r"C:/Users/naman/Desktop/vlc/train/right"



test_up_dir = r"C:\Users\naman\Desktop\vlc\test\up"
test_down_dir = r"C:\Users\naman\Desktop\vlc\test\down"
test_left_dir = r"C:\Users\naman\Desktop\vlc\test\left"
test_nothing_dir = r"C:\Users\naman\Desktop\vlc\train\nothing"
test_right_dir = r"C:\Users\naman\Desktop\vlc\train\right"

num_up_tr = len(os.listdir(train_up_dir))
num_down_tr = len(os.listdir(train_down_dir))
num_left_tr = len(os.listdir(train_left_dir))
num_nothing_tr = len(os.listdir(train_nothing_dir))
num_right_tr = len(os.listdir(train_right_dir))

num_up_test = len(os.listdir(test_up_dir))
num_down_test = len(os.listdir(test_down_dir))
num_left_test = len(os.listdir(test_left_dir))
num_nothing_test = len(os.listdir(test_nothing_dir))
num_right_test = len(os.listdir(test_right_dir))

total_train = num_left_tr + num_down_tr + num_up_tr + num_nothing_tr  
total_test = num_up_test + num_down_test + num_left_test + num_nothing_test

print('total training up images:', num_up_tr)
print('total training down images:', num_down_tr)
print('total training left images:', num_left_tr)
print('total training nothing images:', num_nothing_tr)

print('total test up images:', num_up_test)
print('total test down images:', num_down_test)
print('total test left images:', num_left_test)
print('total test nothing images:', num_nothing_tr)
print("--")
print("Total training images:", total_train)
print("Total test images:", total_test)

IMG_HEIGHT = 150
IMG_WIDTH = 150

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5)

train_data_gen = image_gen_train.flow_from_directory(batch_size=5,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     color_mode = 'grayscale',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=5,
                                                   shuffle = True,
                                                 directory=test_dir,
                                                 color_mode = 'grayscale',
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    epochs=8,
    validation_data=test_data_gen
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(8)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json) 
model.save_weights('model.h5')  









