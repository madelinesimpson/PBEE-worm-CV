from django.shortcuts import render, redirect
import tensorflow as tf
from tensorflow import keras
from wormAppUI import image_preprocessing as ip
from wormAppUI import rectangle_crop as rc
import cv2
import numpy as np
from .models import imageModel
from .forms import imageForm
from PIL import Image

def predict(image):
    questionable_worms = []
    model = keras.models.load_model('./classification_model.h5')
    # Do all of the preprocessing for the image to be analyzed
    # Return processed image and an unprocessed image to draw resulting contours on
    image, contour_image = ip.preprocessImage(image)
    # Get all of the worm contours in the image
    wormblobs, x = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dead_count = 0
    alive_count = 0
    cluster_count = 0
    further_processing_count = 0
    # Process all of the worm contours
    for worm in wormblobs:
        # Isolate the wormblob on its own image
        mask_base = np.zeros((520, 696), np.uint8)
        worm_img = cv2.drawContours(mask_base, [worm], 0, 255, -1)
        # Binarize the worm image
        normalized_worm_img = worm_img / 255.0
        # Find the bounding coordinates of the worm and give it some padding
        xMin, xMax, yMin, yMax = rc.findBoundingCoords(normalized_worm_img)
        xMin, xMax, yMin, yMax = rc.updateBoundingCoords(xMin, xMax, yMin, yMax)
        # Get the skeleton of the worm
        worm_skeleton = rc.findSkeleton(normalized_worm_img, xMin, xMax, yMin, yMax)
        # Non-binarize the skeleton for the endpoints function (idk why it only takes input this way)
        non_binary_skel = worm_skeleton * 255.0
        # Find the endpoints of the skeleton
        endpoints = rc.findEndpoints(non_binary_skel)
        # If the worm has two endpoints, process it as a worm
        if len(endpoints) == 2:
            # Find bounding coords of skeleton to crop into rectangle
            xMin, xMax, yMin, yMax = rc.findBoundingCoords(worm_skeleton)
            # Crop into rectangle
            warped = rc.cropWormIntoRectangle(worm_img, endpoints, xMin, xMax, yMin, yMax)
            # Process the rectangle so that it can be inputted into model
            input = ip.processImageForModel(warped)
            # Make a prediction
            prediction = model.predict(input)
            score = float(prediction[0])
            print(score)
            if score < 0.4:
                alive_count = alive_count + 1
                cv2.drawContours(contour_image, [worm], 0, (0, 255, 0), 1)
            elif score > 0.6:
                dead_count = dead_count + 1
                cv2.drawContours(contour_image, [worm], 0, (255, 0, 0), 1)
            else:
                further_processing_count = further_processing_count + 1
                cv2.drawContours(contour_image, [worm], 0, (150, 0, 255), 1)
                '''
                ind_worm_base = np.zeros((520, 696), np.uint8)
                cv2.drawContours(ind_worm_base, [worm], 0, 255, -1)
                questionable_worms.append(ind_worm_base)
                '''
        elif len(endpoints) == 1:
            alive_count = alive_count + 1
            cv2.drawContours(contour_image, [worm], 0, (0, 255, 0), 1)
        else:
            cluster_count = cluster_count + 1
            cv2.drawContours(contour_image, [worm], 0, (255, 255, 0), 1)

    total_without_clusters = alive_count + dead_count + further_processing_count
    return alive_count, dead_count, cluster_count, total_without_clusters, further_processing_count, contour_image

def clarify(request):
    context = {}
    alive_count = request.session['alive_count']
    dead_count = request.session['dead_count']
    cluster_count = request.session['cluster_count']
    total_without_clusters = request.session['total_without_clusters']
    contour_image_name = request.session['contour_image']
    questionable_worms = request.session['questionable_worms']

    return render(request, "clarify.html", context)


def result(request):
    context = {}

    alive_count, dead_count, cluster_count, total_without_clusters, further_processing_count, contour_array = predict(request.session['image_path'])

    contour_image = Image.fromarray(contour_array)
    contour_image_name = 'contour of ' + request.session['image_name']
    contour_image_path = './contour_images/' + contour_image_name
    contour_image.save(contour_image_path)

    context['alive_count'] = alive_count
    context['dead_count'] = dead_count
    context['cluster_count'] = cluster_count
    context['total_without_clusters'] = total_without_clusters
    context['further_processing_count'] = further_processing_count
    context['contour_image'] = contour_image_name
    '''
    i = 1
    worm_names = []
    for worm in questionable_worms:
        image = Image.fromarray(worm)
        image_name = 'individual worm ' + str(i) + '.png'
        individual_worm_path = './individual_worms/' + image_name
        image.save(individual_worm_path)
        worm_names.append(image_name)
        i += 1
    '''
    return render(request, "result.html", context)


def home(request):
    context = {}
    if request.method == "POST":
        form = imageForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data.get("image_field")
            obj = imageModel.objects.create(
                title="",
                img=img
            )
            image_path = './images/' + str(img)
            request.session['image_name'] = str(img)
            request.session['image_path'] = image_path
            obj.save()

            return redirect(result)
    else:
        form = imageForm()
    context['form'] = form
    return render(request, "index.html", context)