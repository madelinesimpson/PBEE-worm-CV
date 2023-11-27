import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import individual_worm_data as worm
import cluster_worm_data as cluster

image = cv2.imread('./cluster 6.png')

skeleton = image.copy()
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
normalized = skeleton/255.0

xMin, xMax, yMin, yMax = worm.findBoundingCoords(normalized)
xMin, xMax, yMin, yMax = worm.padBounding(xMin, xMax, yMin, yMax)
skeleton = worm.findSkeleton(skeleton, xMin, xMax, yMin, yMax)

plt.figure()
plt.imshow(skeleton)
plt.show()

endpoints = cluster.find_endpoints(skeleton)
print(endpoints)
intersections, new_skeleton = cluster.find_intersections(skeleton, endpoints)
print(intersections)

rows = 1
columns = 2
skeleton = cv2.cvtColor(skeleton.astype('float32'), cv2.COLOR_GRAY2RGB)
new_skeleton = cv2.cvtColor(new_skeleton.astype('float32'), cv2.COLOR_GRAY2RGB)

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
