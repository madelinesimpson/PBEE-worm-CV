import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import individual_worm_data as worm
import cluster_worm_data as cluster
from random import randint

image = cv2.imread('./cluster 2.png')

skeleton = image.copy()
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
normalized = skeleton/255.0

xMin, xMax, yMin, yMax = worm.findBoundingCoords(normalized)
xMin, xMax, yMin, yMax = worm.padBounding(xMin, xMax, yMin, yMax)
skeleton = worm.findSkeleton(skeleton, xMin, xMax, yMin, yMax)

'''
plt.figure()
plt.imshow(skeleton)
plt.show()
'''
endpoints = cluster.find_endpoints(skeleton)
intersections, removed_endpoints, new_endpoints, new_skeleton = cluster.find_intersections(skeleton, endpoints)
intersections = cluster.remove_repeat_intersections(intersections, 8, new_skeleton)


for point in endpoints:
    if point in removed_endpoints:
        endpoints.remove(point)
for point in new_endpoints:
    endpoints.append(point)

branches, undone_points = cluster.get_branches(intersections, endpoints, new_skeleton)
print(undone_points)
other_branches = cluster.get_paths_between_intersections(intersections, undone_points, new_skeleton)


new_skeleton = cv2.cvtColor(new_skeleton.astype('float32'), cv2.COLOR_GRAY2RGB)


for branch in other_branches:
    branches.append(branch)

#branches = cluster.remove_repeat_branches(branches)

for branch in branches:
    for point in branch:
        new_skeleton[point[0]][point[1]][0] = 255
        new_skeleton[point[0]][point[1]][1] = 255
        new_skeleton[point[0]][point[1]][2] = 0

for coord in intersections:
    new_skeleton[coord[0]][coord[1]][0] = 255
    new_skeleton[coord[0]][coord[1]][1] = 0
    new_skeleton[coord[0]][coord[1]][2] = 0
for coord in endpoints:
    new_skeleton[coord[0]][coord[1]][0] = 0
    new_skeleton[coord[0]][coord[1]][1] = 0
    new_skeleton[coord[0]][coord[1]][2] = 255


plt.figure()
plt.imshow(new_skeleton)
plt.show()

'''
fig = plt.figure(figsize=(8, 8))

fig.add_subplot(rows, columns, 1)
plt.imshow(new_skeleton)
fig.add_subplot(rows, columns, 2)
plt.imshow(skeleton)
plt.show()
'''
