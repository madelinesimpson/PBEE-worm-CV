import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#normalised = cv2.imread('/Users/mgsimp2/PycharmProjects/worms/A01.png')

def doUpperThresholding(image, threshold, value):
    """
    doUpperThresholding

    thresholds an image

    @param image: raw image
    @param threshold: threshold value
    @param value: value to threshold by
    @return image: image post thresholding

    """
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if image[i][j] > threshold:
                image[i][j] = value
    return image


def doLowerThresholding(image, threshold, value):
    """
    doLowerThresholding

    thresholds an image

    @param image: raw image
    @param threshold: threshold value
    @param value: value to threshold by
    @return image: image post thresholding

    """
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if image[i][j] < threshold:
                image[i][j] = value
    return image


def createThreshold(normalised):
    """
    createThreshold

    performs various thresholding techniques on a normalised iamge

    @param normalised: normalised image
    @param worm_img: thresholded image

    """

    # make a copy of the normalised
    worm_img = normalised.copy()
    # create a mask
    mask_s = worm_img.copy()

    # blur and threshold the image
    worm_img = cv2.GaussianBlur(worm_img, (9, 9), 0)
    worm_img = cv2.adaptiveThreshold(worm_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                     cv2.THRESH_BINARY, 15, 3)

    # create the mask
    mask_s = doUpperThresholding(mask_s, 20, 255)
    mask_s = doLowerThresholding(mask_s, 20, 0)
    element = np.ones((10, 10), np.uint8)
    mask_s = cv2.erode(mask_s, element, iterations=1)
    # mask = cv2.dilate(mask,element, iterations = 2)

    # invert the image so we can perform operations with the mask
    worm_img = cv2.bitwise_not(worm_img)
    # merge mask and image together
    worm_img = cv2.min(worm_img, mask_s)

    return worm_img


def normaliseImg(raw):
    """
    normaliseImg

    normalise the 16 bit tif file, and fits it in a 8 bit image.

    @param raw: 16 bit image
    @param raw: 8 bit image

    """

    # normalise the 16 bit tif file
    cv2.normalize(raw, raw, 0, 65535, cv2.NORM_MINMAX)
    # fit it back into a 8 bit image so we can work with it
    raw = (raw / 256).astype('uint8')
    return raw


def ThresholdW1(worm_img):
    """
    ThresholdW1

    performs various thresholding techniques on a normalised W1 image

    @param worm_img: normalised image
    @param worm_img: thresholded image

    """

    worm_img = (worm_img / 256).astype('uint8')
    worm_img = cv2.GaussianBlur(worm_img, (5, 5), 0)
    ret, worm_img = cv2.threshold(worm_img, 0, 255, cv2.THRESH_BINARY)
    worm_img = ParticleCleansing(worm_img)
    return worm_img


def ParticleCleansing(image):
    """
    ParticleCleansing

    Searches for particles in an images and removes it using contours.

    @param image: thresholded image
    @param image: sanitised image

    """

    # print("--- Initiating Cleansing ---")
    # find contours in image so we can look for particle
    contours, z = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    worm_count = 0
    for worm in contours:
        area = cv2.contourArea(worm)
        if area > 300:
            # it's a worm, don't mess with it
            worm_count += 1
        else:
            # its most likely a particle, colour it black.
            cv2.drawContours(image, [worm], 0, 0, -1)
    # print("Number of Worm Blobs Detected: " + str(worm_count))
    # print("Number of Particles Detected: " + str(len(contours)-worm_count))
    # print("--- Ending Cleansing ---")
    return image

def neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],	 # P2,P3,P4,P5
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]	# P6,P7,P8,P9

def transitions(neighbour, white):
	count = 0
	n = neighbour + neighbour[0:1]	  # P2, P3, ... , P8, P9, P2
	return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	white = 1
	layer = 0
	skeleton = image.copy()  # deepcopy to protect the original image

	skeleton = skeleton / 255

	changing1 = changing2 = 1		#  the points to be removed (set as 0)
	while changing1 or changing2:   #  iterates until no further changes occur in the image
		layer += 1
		changing1 = []
		rows, columns = skeleton.shape			   # x for rows, y for columns
		for x in range(1, rows - 1):					 # No. of  rows
			for y in range(1, columns - 1):			# No. of columns
				n = neighbours(x, y, skeleton)
				P2,P3,P4,P5,P6,P7,P8,P9 = n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7]

				if skeleton[x][y] == white: # Condition 0: Point P1 in the object regions
					# print(x,y, "is white")
					if 2 <= np.count_nonzero(n) <= 6:	# Condition 1: 2<= N(P1) <= 6
						if transitions(n, white) == 1:	# Condition 2: S(P1)=1
							if P2 * P4 * P6 == 0:	# Condition 3
								if P4 * P6 * P8 == 0:		 # Condition 4
									changing1.append((x,y))
		for x, y in changing1:
			# print("setting ", x,y, "to be 0")
			skeleton[x][y] = 0
		# Step 2
		changing2 = []
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, skeleton)
				if (skeleton[x][y] == white   and		# Condition 0
					2 <= np.count_nonzero(n) <= 6  and	   # Condition 1
					transitions(n, white) == 1 and	  # Condition 2
					P2 * P4 * P8 == 0 and	   # Condition 3
					P2 * P6 * P8 == 0):			# Condition 4
					changing2.append((x,y))
		for x, y in changing2:
			skeleton[x][y] = 0
	skeleton = skeleton * 255
	return skeleton


'''
img_array = image.img_to_array(normalised, dtype="uint8")
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
mask = img_array.copy()
w1 = img_array.copy()
img_array = cv2.GaussianBlur(img_array, (9, 9), 0)
img_array = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                   cv2.THRESH_BINARY, 15, 3)

mask = doUpperThresholding(mask, 20, 255)
mask = doLowerThresholding(mask, 20, 0)
element = np.ones((10, 10), np.uint8)
mask = cv2.erode(mask, element, iterations=1)

worm_img = cv2.bitwise_not(img_array)
# merge mask and image together
worm_img = cv2.min(worm_img, mask)

worm_copy = worm_img.copy()
blobbed = ParticleCleansing(worm_img)

blobbed_copy = blobbed.copy()

skeleton = zhangSuen(blobbed)

w1 = cv2.GaussianBlur(w1, (5,5), 0)
ret, w1 = cv2.threshold(w1, 0,255,cv2.THRESH_BINARY)
w1 = ParticleCleansing(w1)

img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
worm_copy = cv2.cvtColor(worm_copy, cv2.COLOR_BGR2RGB)
blobbed_copy = cv2.cvtColor(blobbed_copy, cv2.COLOR_BGR2RGB)
skeleton = np.float32(skeleton)
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
w1 = cv2.cvtColor(w1, cv2.COLOR_BGR2RGB)

rows, cols = 2,3

plt.subplot(rows,cols,1)
plt.imshow(img_array)
plt.subplot(rows,cols,2)
plt.imshow(mask)
plt.subplot(rows,cols,3)
plt.imshow(worm_copy)
plt.subplot(rows,cols,4)
plt.imshow(blobbed_copy)
plt.subplot(rows,cols,5)
plt.imshow(skeleton)
plt.subplot(rows, cols, 6)
plt.imshow(w1)

plt.show()
'''

'''
tif_image = cv2.imread('/Users/mgsimp2/PycharmProjects/worms/fullplate.tif', cv2.IMREAD_GRAYSCALE)
cv2.normalize(tif_image, tif_image, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
tif_8bit = tif_image.astype('uint8')

tif_8bit = cv2.GaussianBlur(tif_8bit, (9, 9), 0)
tif_8bit = cv2.adaptiveThreshold(tif_8bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                   cv2.THRESH_BINARY, 15, 3)
color_convert = cv2.cvtColor(tif_8bit, cv2.COLOR_BGR2RGB)

plt.figure()

plt.imshow(color_convert)

plt.show()
'''
