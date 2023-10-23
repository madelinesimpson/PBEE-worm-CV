import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

batch_size = 32

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    './rectangles/',
    labels='inferred',
    label_mode="binary",
    batch_size=batch_size,
    validation_split=0.3,
    seed=1969,
    subset="both",
    image_size=(10,100),
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

input_shape = (10,100,3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

epochs = 10

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    validation_data = val_ds,
    shuffle=True,
)

model.save("./classification_model.h5")
