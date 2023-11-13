import numpy as np
import matplotlib.pyplot as plt
import rectangular_crop as rc
import cv2

cluster = cv2.imread('./cluster/cluster 28.png')
cluster = cv2.cvtColor(cluster, cv2.COLOR_BGR2GRAY)
normalized = cluster/255.0

xMin, xMax, yMin, yMax = rc.findBoundingCoords(normalized)

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

xMin, xMax, yMin, yMax = padBounding(xMin, xMax, yMin, yMax)

def findSkeleton(worm, xMin, xMax, yMin, yMax):
    image = cv2.threshold(worm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    white = 1
    layer = 0
    skeleton = image.copy()  # deepcopy to protect the original image

    skeleton = skeleton / 255

    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        print("ZS: Layer ", layer)
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

cluster_skeleton = rc.findSkeleton(cluster, xMin, xMax, yMin, yMax)
#cluster_skeleton = rc.ogSkeleton(cluster)

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


intersection_points, worm = findIntersectPoints(cluster_skeleton, xMin, xMax, yMin, yMax)
color_image = colorImage(worm, xMin, xMax, yMin, yMax)
print(intersection_points)

plt.figure()
plt.imshow(color_image)
plt.show()
