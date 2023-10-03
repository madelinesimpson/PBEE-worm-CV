import cv2
import numpy as np
import csv
import tensorflow
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as image
import seaborn as sns

from PIL import Image

batch_size = 32
labels = []
alive_size = len(os.listdir('./test/alive/'))
dead_size = len(os.listdir('./test/dead/'))
cluster_size = len(os.listdir('./test/cluster/'))

for i in range(0,alive_size):
    labels.append(0)

for i in range(0,dead_size):
    labels.append(1)

for i in range(0,cluster_size):
    labels.append(2)

print(len(labels))
print(labels)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "test",
        validation_split = 0.2,
        subset = "both",
        labels = labels,
        label_mode = "categorical",
        seed = 1337,
        image_size = (590,696),
        batch_size = batch_size,
)
data_augmentation = keras.Sequential(
        [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
        ]
)

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

input_shape = (590,696,3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0/255),

        layers.Conv2D(32,3,padding="same",activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(32,3,padding="same",activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(64,3,padding="same",activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Flatten(),
        layers.Dense(3,activation="softmax")
    ]
)

epochs = 15

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)

model.save("./classification_model.h5")
