import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

#Insert your own local file directory for the image here
file = '/Users/madelinesimpson/PycharmProjects/HTR/microscope.jpeg'

img = image.load_img(file)

img_array = image.img_to_array(img, dtype="uint8")
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

median_blur_image = cv2.medianBlur(img_array, 5)

gaussian_thresh_image = cv2.adaptiveThreshold(median_blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)

gaussian_blur_image = cv2.GaussianBlur(gaussian_thresh_image,(5,5),0)

ret, otsu_thresh_image = cv2.threshold(gaussian_blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu_thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_threshold_blobs_area = 1000
min_threshold_blobs_area = 500

#Draw over contours that are too small
for i in range(1, len(contours)):
    index_level = int(hierarchy[0][i][1])
    if (index_level<=i):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area<=min_threshold_blobs_area or area>=max_threshold_blobs_area):
            cv2.drawContours(otsu_thresh_image, [cnt], -1, 255, -1, 1)

rgb = cv2.cvtColor(otsu_thresh_image, cv2.COLOR_BGR2RGB)




plt.figure(figsize=(10,10))
plt.imshow(rgb)
plt.show()
