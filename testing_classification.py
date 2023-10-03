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

model = keras.models.load_model('./classification_model.h5')
img = keras.utils.load_img("./cluster.png")
img = tf.image.resize(img, (590,696))
img_array = keras.utils.img_to_array(img)
img_array = img_array.astype("float32")/255
img_array = np.reshape(img_array, (1,590,696,3))

predictions = model.predict(img_array)
print(predictions)
categories = ['alive', 'dead', 'cluster']
print(categories[predictions.argmax()])
