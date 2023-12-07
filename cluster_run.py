import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from numpy import ndarray

import individual_worm_data as worm
import cluster_worm_data as cluster
from random import randint

image = cv2.imread('./cluster 2.png')

contour_image = image.copy()
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

skeleton = image.copy()
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
normalized = skeleton/255.0

xMin, xMax, yMin, yMax = worm.findBoundingCoords(normalized)
xMin, xMax, yMin, yMax = worm.padBounding(xMin, xMax, yMin, yMax)
skeleton = worm.findSkeleton(skeleton, xMin, xMax, yMin, yMax)

#Getting intersections and endpoints
endpoints = cluster.find_endpoints(skeleton)
intersections, removed_endpoints, new_endpoints, new_skeleton = cluster.find_intersections(skeleton, endpoints)
intersections = cluster.remove_repeat_intersections(intersections, 8, new_skeleton)

#Removing endpoints that have been marked to be removed (false endpoints)
for point in endpoints:
    if point in removed_endpoints:
        endpoints.remove(point)
for point in new_endpoints:
    endpoints.append(point)

#Getting branches from endpoint to intersection and branches between intersections, then removing repeat branches
branches, undone_points = cluster.get_branches(intersections, endpoints, new_skeleton)
other_branches = cluster.get_paths_between_intersections(intersections, undone_points, new_skeleton)
for branch in other_branches:
    branches.append(branch)

#unique_branches = cluster.remove_repeat_branches(branches)
counter = 0
radius_calculation_image = np.zeros((520,696), dtype=np.uint8)
radius_calculation_image = np.expand_dims(radius_calculation_image, -1)
cv2.drawContours(radius_calculation_image, contours, 0, (255, 255, 255), 1)
cp_image = radius_calculation_image.copy()
cp_image = cv2.cvtColor(cp_image, cv2.COLOR_GRAY2RGB)

for branch in branches:
    branch_image = np.zeros((520,696), dtype=np.uint8)
    path = branch['path']
    for point in path:
        branch_image[point[0]][point[1]] = 255.0
    normalized_branch_image = branch_image/255.0
    endpoints = branch['endpoints']
    new_xMin, new_xMax, new_yMin, new_yMax = worm.findBoundingCoords(normalized_branch_image)
    new_xMin, new_xMax, new_yMin, new_yMax = worm.padBounding(new_xMin, new_xMax, new_yMin, new_yMax)
    worm_orientation = worm.findImageOrientation(branch_image, new_xMin, new_xMax, new_yMin, new_yMax, endpoints)
    skeleton_array = []
    for x in range(yMin+1, yMax-1):
        for y in range(xMin+1, xMax-1):
            if branch_image[x][y]!=0:
                skeleton_array.append((x,y))

    sorted_skeleton_array = worm.sort_skeleton_array(skeleton_array, worm_orientation)
    control_points = worm.get_control_points(sorted_skeleton_array)

    for coord in control_points:
        cp_image[coord[0]][coord[1]][0] = 0
        cp_image[coord[0]][coord[1]][1] = 255
        cp_image[coord[0]][coord[1]][2] = 0

    '''
    radii = worm.circles(control_points, image)
    for data in radii:
        cv2.circle(cp_image, data[0], data[1], (255, 0, 0), 1)
    '''

    counter+=1
    print('done')

new_skeleton = cv2.cvtColor(new_skeleton.astype('float32'), cv2.COLOR_GRAY2RGB)

plt.figure()
plt.imshow(cp_image)
plt.show()



#Include the endpoints of each branch with the branch, the current system doesn't work because some endpoints have more than one neighbour
