from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import io
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt


data_dir = r'C:\Data Scients\Hackatons\AI_Hackathon_17_02_2024\ML Datasets\main_dataset'

image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory(
    # 'C:\\Data Scients\\Hackatons\\AI_Hackathon_17_02_2024\\ML Datasets\\testdataset'
    'C:\Data Scients\Hackatons\AI_Hackathon_17_02_2024\ML Datasets\main_dataset'
)

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

data = data.map(lambda x, y: (x/255, y))

scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val,
                 callbacks=[tensorboard_callback])


model.save(
    'C:\Data Scients\Hackatons\AI_Hackathon_17_02_2024\ML model\imageclassifier.h5')
