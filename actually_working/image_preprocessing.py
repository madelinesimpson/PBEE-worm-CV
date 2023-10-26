import cv2
import numpy as np

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

def particleCleansing(image):
    contours,x= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    worm_count = 0
    for worm in contours:
        area = cv2.contourArea(worm)
        if area > 300:
            worm_count += 1
        else:
            cv2.drawContours(image,[worm],0,0,-1)
    return image

def preprocessImage(image_directory):
    image = cv2.imread(image_directory)
    contour_image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = createThreshold(image)
    image = particleCleansing(image)

    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    return image, contour_image

def processImageForModel(image):
    input = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    input = cv2.resize(input, (100, 10))
    input = np.expand_dims(input, 0)

    return input
