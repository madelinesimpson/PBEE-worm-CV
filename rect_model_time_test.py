import cv2
import skeleton_test
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

def doUpperThresholding(image, threshold, value):
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if image[i][j] > threshold:
                image[i][j] = value
    return image
def doLowerThresholding(image, threshold, value):
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if image[i][j] < threshold:
                image[i][j] = value
    return image
def createThreshold(normalised):
    worm_img = normalised.copy()
    mask_s = worm_img.copy()

    worm_img = cv2.GaussianBlur(worm_img, (9, 9), 0)
    worm_img = cv2.adaptiveThreshold(worm_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                     cv2.THRESH_BINARY, 15, 3)

    mask_s = doUpperThresholding(mask_s, 20, 255)
    mask_s = doLowerThresholding(mask_s, 20, 0)
    element = np.ones((10, 10), np.uint8)
    mask_s = cv2.erode(mask_s, element, iterations=1)

    worm_img = cv2.bitwise_not(worm_img)
    worm_img = cv2.min(worm_img, mask_s)

    return worm_img
def ParticleCleansing(image):
    contours,x= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    worm_count = 0
    for worm in contours:
        area = cv2.contourArea(worm)
        if area > 300:
            worm_count += 1
        else:
            cv2.drawContours(image,[worm],0,0,-1)
    return image

model = keras.models.load_model('./classification_model.h5')
tic = time.perf_counter()
image = cv2.imread('./B05.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = createThreshold(image)
image = ParticleCleansing(image)
wormblobs, x = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = cv2.imread('./B05.png')
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

dead_count = 0
alive_count = 0
cluster_count = 0

for worm in wormblobs:
    mask_base = np.zeros((520,696), np.uint8)
    worm_img = cv2.drawContours(mask_base, [worm], 0, 255, -1)
    skeleton_image = skeleton_test.zhangSuen(worm_img).astype("float32")
    normalized_skeleton_image = skeleton_image / 255.0
    endpoints = skeleton_test.findEndpoints(skeleton_image)
    if len(endpoints) == 2:
        xMin, xMax, yMin, yMax = skeleton_test.findBoundingCoords(normalized_skeleton_image)
        warped = skeleton_test.rectangleCrop(worm_img, endpoints, xMin, xMax, yMin, yMax)
        input = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(input, (100,10))
        input = np.expand_dims(input, 0)
        print(input.shape)
        prediction = model.predict(input)
        score = float(prediction[0])
        print(score)
        if score<0.5:
            print('alive worm found')
            alive_count = alive_count + 1
            cv2.drawContours(contour_image, [worm], 0,(0,255,0), -1)
        else:
            print('dead worm found')
            dead_count = dead_count + 1
            cv2.drawContours(contour_image, [worm], 0, (255, 0, 0), -1)
    elif len(endpoints) == 1:
        print("circle worm found")
        alive_count = alive_count + 1
    else:
        print("cluster found")
        cluster_count = cluster_count + 1
        cv2.drawContours(contour_image, [worm], 0, (255, 255, 0), -1)

print("alive count: ", alive_count)
print("dead count: ", dead_count)
print("cluster_count: ", cluster_count)
print("total without clusters: ", alive_count+dead_count)
toc = time.perf_counter()

print("that took ", toc-tic, " seconds")

plt.figure()
plt.imshow(contour_image)
plt.show()
