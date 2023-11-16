import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def findImageOrientation(skeleton, xMin, xMax, yMin, yMax):
    orientation = ""
    (rows, cols) = np.nonzero(skeleton)
    endpoints = []
    for (r, c) in zip(rows, cols):
        counter = countNeighbouringPixels(skeleton, r, c)
        if counter == 1:
            endpoints.append((r, c))
    xDist = xMax - xMin
    yDist = yMax - yMin
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

def sortSkeletonArray(skeleton_array, orientation):
    if orientation == "vertical":
        skeleton_array.sort(key=lambda unsorted_skeleton_array: unsorted_skeleton_array[0])
    else:
        skeleton_array.sort(key=lambda unsorted_skeleton_array: unsorted_skeleton_array[1])
    return skeleton_array

def findRadii(data_image, orientation, sorted_skeleton_array, control_points, xMin, xMax, yMin, yMax):
    radii = []
    for coord in control_points:
        points = []
        skel_index = sorted_skeleton_array.index(coord)
        before = sorted_skeleton_array[skel_index-1]
        after = sorted_skeleton_array[skel_index+ 1]
        if orientation == "vertical":
            if before[0] == coord[0] or after[0] == coord[0]:
                for y in range(coord[1]-20, coord[1]+20):
                    if data_image[coord[0]][y] != 0:
                        points.append(y)
                radius = 0
                for i in range(0, len(points) - 1):
                    value = abs(points[i] - points[i + 1])
                    if value > radius:
                        radius = value
                if radius==0:
                    for y in range(coord[1] - 20, coord[1] + 20):
                        if data_image[coord[0]][y] != 0:
                            points.append(y)
                    radius = 0
                    for i in range(0, len(points) - 1):
                        value = abs(points[i] - points[i + 1])
                        if value > radius:
                            radius = value
                radii.append((coord, radius))
            else:
                for x in range(coord[0]-20, coord[0]+20):
                    if data_image[x][coord[1]] != 0:
                        points.append(x)
                radius = 0
                for i in range(0, len(points) - 1):
                    value = abs(points[i] - points[i + 1])
                    if value > radius:
                        radius = value
                if radius==0:
                    for x in range(coord[0] - 20, coord[0] + 20):
                        if data_image[x][coord[1]] != 0:
                            points.append(x)
                    radius = 0
                    for i in range(0, len(points) - 1):
                        value = abs(points[i] - points[i + 1])
                        if value > radius:
                            radius = value
                radii.append((coord, radius))
        else:
            if before[1] == coord[1] or after[1] == coord[1]:
                for y in range(coord[1]-20, coord[1]+20):
                    if data_image[coord[0]][y] != 0:
                        points.append(y)
                radius = 0
                for i in range(0, len(points)-1):
                    value = abs(points[i] - points[i+1])
                    if value>radius:
                        radius = value
                if radius==0:
                    for x in range(coord[0] - 20, coord[0] + 20):
                        if data_image[x][coord[1]] != 0:
                            points.append(x)
                    radius = 0
                    for i in range(0, len(points) - 1):
                        value = abs(points[i] - points[i + 1])
                        if value > radius:
                            radius = value
                radii.append((coord, radius))
            else:
                for x in range(coord[0]-20, coord[0]+20):
                    if data_image[x][coord[1]] != 0:
                        points.append(x)
                radius = 0
                for i in range(0, len(points) - 1):
                    value = abs(points[i] - points[i + 1])
                    if value > radius:
                        radius = value
                if radius==0:
                    for y in range(coord[1] - 20, coord[1] + 20):
                        if data_image[coord[0]][y] != 0:
                            points.append(y)
                    radius = 0
                    for i in range(0, len(points) - 1):
                        value = abs(points[i] - points[i + 1])
                        if value > radius:
                            radius = value
                radii.append((coord, radius))
    return radii
