import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def findBoundingCoords(normalized_skeleton_image):
    xMin = 1000
    yMin = 1000
    xMax = -1000
    yMax = -1000
    y = 0
    for row in normalized_skeleton_image:
        x = 0
        for pixel in row:
            if pixel == 1:
                if x < xMin:
                    xMin = x
                    x += 1
                elif x > xMax:
                    xMax = x
                    x += 1
                elif y < yMin:
                    yMin = y
                    x += 1
                elif y > yMax:
                    yMax = y
                    x += 1
                else:
                    x += 1
                    continue
            else:
                x += 1
                continue
        y += 1
    return xMin, xMax, yMin, yMax

def padBounding(xMin,xMax,yMin,yMax):
    if xMin>=10:
        xMin-=10
    else:
        xMin=0
    if xMax<686:
        xMax+=10
    else:
        xMax=695

    if yMin>=10:
        yMin-=10
    else:
        yMin=0
    if yMax<510:
        yMax+=10
    else:
        yMax=519
    return xMin, xMax, yMin, yMax

def transitions(neighbour, white):
    count = 0
    n = neighbour + neighbour[0:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def get_neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]

def findSkeleton(worm, xMin, xMax, yMin, yMax):
    image = cv2.threshold(worm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    white = 1
    layer = 0
    skeleton = image.copy()  # deepcopy to protect the original image

    skeleton = skeleton / 255

    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        layer += 1
        changing1 = []
        for x in range(yMin + 1, yMax - 1):  # No. of  rows
            for y in range(xMin + 1, xMax - 1):  # No. of columns
                n = get_neighbours(x, y, skeleton)
                P2, P3, P4, P5, P6, P7, P8, P9 = n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]
                if skeleton[x][y] == white:  # Condition 0: Point P1 in the object regions
                    # print(x,y, "is white")
                    if 2 <= np.count_nonzero(n) <= 6:  # Condition 1: 2<= N(P1) <= 6
                        if transitions(n, white) == 1:  # Condition 2: S(P1)=1
                            if P2 * P4 * P6 == 0:  # Condition 3
                                if P4 * P6 * P8 == 0:  # Condition 4
                                    changing1.append((x, y))
        for x, y in changing1:
            # print("setting ", x,y, "to be 0")
            skeleton[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(yMin + 1, yMax - 1):  # No. of  rows
            for y in range(xMin + 1, xMax - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = get_neighbours(x, y, skeleton)
                if (skeleton[x][y] == white and  # Condition 0
                        2 <= np.count_nonzero(n) <= 6 and  # Condition 1
                        transitions(n, white) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            skeleton[x][y] = 0
    skeleton = skeleton * 255
    return skeleton

def findIntersectPoints(skeleton, xMin, xMax, yMin, yMax):
    intersect_points = []
    for x in range(yMin+1, yMax-1):
        for y in range(xMin+1, xMax-1):
                if skeleton[x][y]!=0:
                    n = rc.get_neighbours(x,y,skeleton)
                    counter = sum(n)/255
                    if counter >= 4:
                        intersect_points.append((x,y))
                        skeleton[x][y]=69
    return intersect_points, skeleton

def colorImage(worm, xMin, xMax, yMin, yMax):
    color_image = np.expand_dims(worm, -1)
    color_image = np.float32(color_image)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
    for x in range(xMin+1, xMax-1):
        for y in range(yMin+1, yMax-1):
            if worm[y][x]==69:
                color_image[y][x][0] = 255
                color_image[y][x][1] = 0
                color_image[y][x][2] = 0
    return color_image

def countNeighbouringPixels(skeleton, x, y):
    neighbours = get_neighbours(x, y, skeleton)
    return sum(neighbours) / 255

def findImageOrientation(skeleton, xMin, xMax, yMin, yMax, endpoints):
    orientation = ""
    '''
    (rows, cols) = np.nonzero(skeleton)
    endpoints = []
    for (r, c) in zip(rows, cols):
        counter = countNeighbouringPixels(skeleton, r, c)
        if counter == 1:
            endpoints.append((r, c))
    '''
    xDist = xMax - xMin
    yDist = yMax - yMin
    print(len(endpoints))
    endpoint_x1 = endpoints[0][1]
    endpoint_x2 = endpoints[1][1]
    endpoint_y1 = endpoints[0][0]
    endpoint_y2 = endpoints[1][0]
    vert = False

    if endpoint_x1 == endpoint_x2:
        vert = True
    else:
        endpoint_line_slope = (endpoint_y2 - endpoint_y1) / (endpoint_x2 - endpoint_x1)

    if vert==True:
        orientation = "vertical"
    elif ((abs(endpoint_line_slope) < 1) or (abs(endpoint_line_slope == 1) and xDist > yDist)):
        orientation = "horizontal"
    else:
        orientation = "vertical"

    return orientation

def sort_skeleton_array(skeleton_array, orientation):
    if orientation == "vertical":
        skeleton_array.sort(key=lambda unsorted_skeleton_array: unsorted_skeleton_array[0])
    else:
        skeleton_array.sort(key=lambda unsorted_skeleton_array: unsorted_skeleton_array[1])
    return skeleton_array

def get_control_points(sorted_skeleton_array):
    num_control_points = 19
    control_points = []
    spacing = len(sorted_skeleton_array)//num_control_points
    i=0
    for coord in sorted_skeleton_array:
        if i % spacing == 0:
            control_points.append(coord)
        i += 1
    return control_points

def circles(control_points, grayscale_worm):
    radii = []
    grayscale_worm = grayscale_worm/255.0
    for coord in control_points:
        r=1
        radius_done = False
        compare_image = grayscale_worm.copy()
        while radius_done!=True:
            cv2.circle(compare_image, (coord[1], coord[0]), r, 1, 1)
            comparison = cv2.bitwise_xor(compare_image, grayscale_worm)
            value = np.count_nonzero(comparison)
            if value!=0:
                radius_done = True
            else:
                r+=1
        radii.append([(coord[1], coord[0]), r])
    return radii

def normalize_points(control_points):
    index = (len(control_points))//2
    translation_factor = control_points[index]
    translate_y = translation_factor[1]
    translate_x = translation_factor[0]
    for i in range(0, len(control_points)):
        x = control_points[i][0] - translate_x
        y = control_points[i][1] - translate_y
        control_points[i] = (x,y)
    return control_points

def find_rotation_factor(normalized_control_points):
    min_eucledian_dist = 1000000
    rotation_factor = 0
    theta = 0
    while theta<math.pi:
        dist = 0
        for coord in normalized_control_points:
            x = coord[0] * math.cos(theta) - coord[1] * math.sin(theta)
            squared = x*x
            dist+=squared
        if dist<min_eucledian_dist:
            min_eucledian_dist = dist
            rotation_factor = theta
        theta = theta + (math.pi/180)
    return min_eucledian_dist, rotation_factor

def rotate_points(normalized_control_points, rotation_factor):
    for i in range(0, len(normalized_control_points)):
        x = normalized_control_points[i][0] * math.cos(rotation_factor) - normalized_control_points[i][1] * math.sin(rotation_factor)
        y = normalized_control_points[i][1] * math.cos(rotation_factor) + normalized_control_points[i][0] * math.sin(rotation_factor)
        x = round(x, 3)
        y = round(y, 3)
        normalized_control_points[i] = (x,y)

    return normalized_control_points

def mirror_worm(rotated_control_points):
    mirrored_coords = []
    for coord in rotated_control_points:
        x = coord[0]
        y = coord[1]
        mirrored_coords.append((-x,y))
    return mirrored_coords
