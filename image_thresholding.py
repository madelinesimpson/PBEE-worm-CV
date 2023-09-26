import numpy
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import tensorflow as tf
import torch
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import pickle

'''
Sam Setup
'''
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=400,  # Requires open-cv to run post-processing
)

#Insert your specific path to the image here
file = '/Users/madelinesimpson/PycharmProjects/HTR/testworm.jpg'

'''
Process the image to be black and white and remove as many unwanted blobs as possible
'''
def preprocess_image(file):
    img = image.load_img(file)

    #Make the image an array and convert to grayscale
    img_array = image.img_to_array(img, dtype="uint8")
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    #Get those worms to pop
    median_blur_image = cv2.medianBlur(img_array, 5)
    gaussian_thresh_image = cv2.adaptiveThreshold(median_blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                      cv2.THRESH_BINARY,11,2)
    gaussian_blur_image = cv2.GaussianBlur(gaussian_thresh_image,(5,5),0)

    ret, otsu_thresh_image = cv2.threshold(gaussian_blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(otsu_thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Remove small blobs from the background
    threshold_blobs_area = 500
    for i in range(1, len(contours)):
        index_level = int(hierarchy[0][i][1])
        if (index_level<=i):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area<=threshold_blobs_area):
                cv2.drawContours(otsu_thresh_image, [cnt], -1, 255, -1, 1)

    #Convert image to RGB
    rgb = cv2.cvtColor(otsu_thresh_image, cv2.COLOR_BGR2RGB)
    return rgb

'''
Let SAM generate the masks
'''
processed_image = preprocess_image(file)
masks = mask_generator_2.generate(processed_image)

with open('/Users/madelinesimpson/PycharmProjects/HTR/masks.pkl', 'wb') as fp:
    pickle.dump(masks, fp)
    print('masks saved successfully to file')
