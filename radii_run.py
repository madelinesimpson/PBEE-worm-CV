import cv2
import numpy as np
import matplotlib.pyplot as plt
import individual_worm_data as worm

image = cv2.imread('./alive 5.png')

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

print(worm_orientation)
cp_dist = 6
control_points = []

cp_image = radius_calculation_image.copy()
cp_image = cv2.cvtColor(cp_image, cv2.COLOR_GRAY2RGB)

i=1
for coord in sorted_skeleton_array:
    if i%cp_dist==0:
        cp_image[coord[0]][coord[1]][0] = 255
        cp_image[coord[0]][coord[1]][1] = 255
        cp_image[coord[0]][coord[1]][2] = 255
        control_points.append(coord)
    i+=1

radii = worm.findRadii(radius_calculation_image, worm_orientation, sorted_skeleton_array, control_points, xMin, xMax, yMin, yMax)
print(radii)

#skeleton /= 255.0
#radius_calculation_image = radius_calculation_image / 255.0

#both = cv2.bitwise_or(skeleton, radius_calculation_image)

plt.figure()
plt.imshow(cp_image)
plt.show()
