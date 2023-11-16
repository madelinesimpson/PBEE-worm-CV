import cv2
import numpy as np
import matplotlib.pyplot as plt
import individual_worm_data as worm

image = cv2.imread('./alive 4.png')

skeleton = image.copy()
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
normalized = skeleton/255.0

xMin, xMax, yMin, yMax = worm.findBoundingCoords(normalized)
xMin, xMax, yMin, yMax = worm.padBounding(xMin, xMax, yMin, yMax)
skeleton = worm.findSkeleton(skeleton, xMin, xMax, yMin, yMax)
worm_orientation = worm.findImageOrientation(skeleton, xMin, xMax, yMin, yMax)
#cluster_skeleton = rc.ogSkeleton(cluster)

skeleton_array = []
for x in range(yMin+1, yMax-1):
    for y in range(xMin+1, xMax-1):
        if skeleton[x][y]!=0:
            skeleton_array.append((x,y))

sorted_skeleton_array = worm.sortSkeletonArray(skeleton_array, worm_orientation)

contour_image = image.copy()
contour_image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

radius_calculation_image = np.zeros((520,696), dtype=np.uint8)
radius_calculation_image = np.expand_dims(radius_calculation_image, -1)
cv2.drawContours(radius_calculation_image, contours, 0, (255, 255, 255), 1)
#radius_calculation_image = cv2.cvtColor(radius_calculation_image, cv2.COLOR_GRAY2RGB)

cp_dist = 8
control_points = []

cp_image = radius_calculation_image.copy()
cp_image = cv2.cvtColor(cp_image, cv2.COLOR_GRAY2RGB)

i=0
for coord in sorted_skeleton_array:
    if i%cp_dist==0:
        cp_image[coord[0]][coord[1]][0] = 255
        cp_image[coord[0]][coord[1]][1] = 255
        cp_image[coord[0]][coord[1]][2] = 255
        control_points.append(coord)
    i+=1

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for coord in control_points:
    cp_image[coord[0]][coord[1]][0] = 0
    cp_image[coord[0]][coord[1]][1] = 255
    cp_image[coord[0]][coord[1]][2] = 0

radii = worm.circles(control_points, image)
for data in radii:
    cv2.circle(cp_image, data[0], data[1], (255,0,0),1)

swapped = control_points
for i in range(0,len(swapped)):
    x = swapped[i][1]
    y = swapped[i][0]
    swapped[i] = (x,y)

normalized_points = worm.normalize_points(control_points)
min_eucledian_dist, rotation_factor = worm.find_rotation_factor(normalized_points)
rotated_points = worm.rotate_points(normalized_points, rotation_factor)
print(rotated_points)

plt.figure()
plt.imshow(cp_image)
plt.show()
