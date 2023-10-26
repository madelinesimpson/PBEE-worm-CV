import image_preproccesing as edit
import rectangular_crop as crop
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('./classification_model.h5')

#Do all of the preprocessing for the image to be analyzed
#Return processed image and an unprocessed image to draw resulting contours on
image, contour_image = edit.preprocessImage('./A14.png')

#Get all of the worm contours in the image
wormblobs, x = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

dead_count = 0
alive_count = 0
cluster_count = 0

#Process all of the worm contours
for worm in wormblobs:
    #Isolate the wormblob on its own image
    mask_base = np.zeros((520,696), np.uint8)
    worm_img = cv2.drawContours(mask_base, [worm], 0, 255, -1)

    #Binarize the worm image
    normalized_worm_img = worm_img/255.0

    #Find the bounding coordinates of the worm and give it some padding
    xMin, xMax, yMin, yMax = crop.findBoundingCoords(normalized_worm_img)
    xMin, xMax, yMin, yMax = crop.updateBoundingCoords(xMin, xMax, yMin, yMax)

    #Get the skeleton of the worm
    worm_skeleton = crop.findSkeleton(normalized_worm_img, xMin, xMax, yMin, yMax)
    plt.figure()
    plt.imshow(worm_skeleton)
    plt.show()

    #Non-binarize the skeleton for the endpoints function (idk why it only takes input this way)
    non_binary_skel = worm_skeleton * 255.0
    #Find the endpoints of the skeleton
    endpoints = crop.findEndpoints(non_binary_skel)

    #If the worm has two endpoints, process it as a worm
    if len(endpoints) == 2:
        #Find bounding coords of skeleton to crop into rectangle
        xMin, xMax, yMin, yMax = crop.findBoundingCoords(worm_skeleton)
        #Crop into rectangle
        warped = crop.cropWormIntoRectangle(worm_img, endpoints, xMin, xMax, yMin, yMax)

        #Process the rectangle so that it can be inputted into model
        input = edit.processImageForModel(warped)

        #Make a prediction
        prediction = model.predict(input)
        score = float(prediction[0])

        print(score)
        if score<0.5:
            print('alive worm found')
            alive_count = alive_count + 1
            cv2.drawContours(contour_image, [worm], 0,(0,255,0), 1)
        else:
            print('dead worm found')
            dead_count = dead_count + 1
            cv2.drawContours(contour_image, [worm], 0, (255, 0, 0), 1)
    elif len(endpoints) == 1:
        print("circle worm found")
        alive_count = alive_count + 1
    else:
        print("cluster found")
        cluster_count = cluster_count + 1
        cv2.drawContours(contour_image, [worm], 0, (255, 255, 0), 1)

print("alive count: ", alive_count)
print("dead count: ", dead_count)
print("cluster_count: ", cluster_count)
print("total without clusters: ", alive_count+dead_count)

plt.figure()
plt.imshow(contour_image)
plt.show()
