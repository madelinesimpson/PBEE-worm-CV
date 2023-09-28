import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import main as m
import matplotlib.pyplot as plt

normalised = cv2.imread('/Users/mgsimp2/PycharmProjects/worms/A01.png')

img_array = image.img_to_array(normalised, dtype="uint8")
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

mask = img_array.copy()
w1 = img_array.copy()

img_array = cv2.GaussianBlur(img_array, (9, 9), 0)
img_array = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                   cv2.THRESH_BINARY, 15, 3)

mask = m.doUpperThresholding(mask, 20, 255)
mask = m.doLowerThresholding(mask, 20, 0)
element = np.ones((10, 10), np.uint8)
mask = cv2.erode(mask, element, iterations=1)

worm_img = cv2.bitwise_not(img_array)
worm_img = cv2.min(worm_img, mask)

blobbed = m.ParticleCleansing(worm_img)

#skeleton = m.zhangSuen(blobbed)

w1 = (w1/256).astype('uint8')
w1 = cv2.GaussianBlur(w1, (5,5), 0)
ret, w1 = cv2.threshold(w1, 0,255,cv2.THRESH_BINARY)
w1 = m.ParticleCleansing(w1)

img = cv2.bitwise_or(blobbed,w1)

wormblobs, y = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blobbed = cv2.cvtColor(blobbed,cv2.COLOR_BGR2RGB)
w1 = cv2.cvtColor(w1,cv2.COLOR_BGR2RGB)

img1 = img.copy()
'''
one_blob = wormblobs[1].reshape(-1,2)
for (x,y) in one_blob:
    cv2.circle(img1, (x,y), 1, (255,0,0), 3)
    '''

one_blob = wormblobs[1]
mask_base = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask_base, [one_blob], 0,255,-1)

img1 = cv2.bitwise_and(img1,img1,mask=mask_base)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

rows,cols = 1,4

plt.figure()
plt.subplot(rows,cols,1)
plt.imshow(w1)
plt.subplot(rows,cols,2)
plt.imshow(blobbed)
plt.subplot(rows,cols,3)
plt.imshow(img)
plt.subplot(rows,cols,4)
plt.imshow(img1)
plt.show()
