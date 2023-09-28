import cv2
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import main as m
import matplotlib.pyplot as plt

with open('/Users/mgsimp2/Desktop/testdataset.csv', 'r') as f2:
        data = f2.read()
        #print(data)

print(data[0])
