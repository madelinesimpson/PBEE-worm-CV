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

test_ds = keras.utils.image_dataset_from_directory(
    './rectanglestest/',
    labels='inferred',
    label_mode="binary",
    batch_size=32,
    image_size=(10, 100),
)

model = keras.models.load_model('./classification_model.h5')

score = model.evaluate(test_ds, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
