import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sympy.solvers import solve
from sympy import Symbol

def neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],	 # P2,P3,P4,P5
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]	# P6,P7,P8,P9

def get_neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],	 # P2,P3,P4,P5
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]	# P6,P7,P8,P9

def transitions(neighbour, white):
	count = 0
	n = neighbour + neighbour[0:1]	  # P2, P3, ... , P8, P9, P2
	return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	white = 1
	layer = 0
	skeleton = image.copy()  # deepcopy to protect the original image

	skeleton = skeleton / 255

	changing1 = changing2 = 1		#  the points to be removed (set as 0)
	while changing1 or changing2:   #  iterates until no further changes occur in the image
		layer += 1
		#print("ZS: Layer " , layer)
		changing1 = []
		rows, columns = skeleton.shape			   # x for rows, y for columns
		for x in range(1, rows - 1):					 # No. of  rows
			for y in range(1, columns - 1):			# No. of columns
				n = neighbours(x, y, skeleton)
				P2,P3,P4,P5,P6,P7,P8,P9 = n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7]

				if skeleton[x][y] == white: # Condition 0: Point P1 in the object regions
					# print(x,y, "is white")
					if 2 <= np.count_nonzero(n) <= 6:	# Condition 1: 2<= N(P1) <= 6
						if transitions(n, white) == 1:	# Condition 2: S(P1)=1
							if P2 * P4 * P6 == 0:	# Condition 3
								if P4 * P6 * P8 == 0:		 # Condition 4
									changing1.append((x,y))
		for x, y in changing1:
			# print("setting ", x,y, "to be 0")
			skeleton[x][y] = 0
		# Step 2
		changing2 = []
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, skeleton)
				if (skeleton[x][y] == white   and		# Condition 0
					2 <= np.count_nonzero(n) <= 6  and	   # Condition 1
					transitions(n, white) == 1 and	  # Condition 2
					P2 * P4 * P8 == 0 and	   # Condition 3
					P2 * P6 * P8 == 0):			# Condition 4
					changing2.append((x,y))
		for x, y in changing2:
			skeleton[x][y] = 0
	skeleton = skeleton * 255
	return skeleton

def findEndpoints(skeleton):
	(rows,cols) = np.nonzero(skeleton)
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter == 1:
			skel_coords.append((r,c))
	return skel_coords

def countNeighbouringPixels(skeleton, x, y):
    neighbours = get_neighbours(x,y,skeleton)
    return sum(neighbours) / 255

def countSkeletonPixels(skeleton):
    pixels = 0
    for i in range(0, len(skeleton)):
        for j in range(0, len(skeleton[0])):
            if skeleton[i][j] != 0:
                pixels += 1
    return pixels

def findBoundingCoords(normalized_skeleton_image):
	xMin=1000
	yMin=1000
	xMax=-1000
	yMax=-1000
	y=0
	for row in normalized_skeleton_image:
		x=0
		for pixel in row:
			if pixel==1:
				if x<xMin:
					xMin=x
					x+=1
				elif x>xMax:
					xMax=x
					x+=1
				elif y<yMin:
					yMin=y
					x+=1
				elif y>yMax:
					yMax=y
					x+=1
				else:
					x+=1
					continue
			else:
				x+=1
				continue
		y+=1
	return xMin,xMax,yMin,yMax

def findRectCoords(image, endpoints, xMin, xMax, yMin, yMax):
	xDist = xMax-xMin
	yDist = yMax-yMin
	endpoint_line_slope = (endpoints[1][0]-endpoints[0][0])/(endpoints[1][1]-endpoints[0][1])
	endpoint_line_y_intercept = endpoints[0][0] - (endpoint_line_slope*endpoints[0][1])
	if endpoint_line_slope==0:
		perpendicular_slope=0
	else:
		perpendicular_slope = -(1 / endpoint_line_slope)

	corners = []
	if (xDist>yDist):
		bottomY = (endpoint_line_slope*xMin) + endpoint_line_y_intercept
		topY = (endpoint_line_slope*xMax) + endpoint_line_y_intercept

		perpendicular_top_y_intercept = topY - (perpendicular_slope*xMax)
		perpendicular_bottom_y_intercept = bottomY - (perpendicular_slope*xMin)

		x = Symbol('x')
		top_result = solve(((x - xMax) * (x - xMax)) + (((perpendicular_slope * x + perpendicular_top_y_intercept) - topY) * ((perpendicular_slope * x + perpendicular_top_y_intercept) - topY)) - (5 * 5), x)
		bottom_result = solve(((x - xMin) * (x - xMin)) + (((perpendicular_slope * x + perpendicular_bottom_y_intercept) - bottomY) * ((perpendicular_slope * x + perpendicular_bottom_y_intercept) - bottomY)) - (5 * 5), x)
		print(bottom_result)
		top_negative_x = top_result[0].evalf()
		top_positive_x = top_result[1].evalf()
		bottom_negative_x = bottom_result[0].evalf()
		bottom_positive_x = bottom_result[1].evalf()


		top_negative_y = (top_negative_x * perpendicular_slope) + perpendicular_top_y_intercept
		top_positive_y = (top_positive_x * perpendicular_slope) + perpendicular_top_y_intercept
		bottom_negative_y = (bottom_negative_x * perpendicular_slope) + perpendicular_bottom_y_intercept
		bottom_positive_y = (bottom_positive_x * perpendicular_slope) + perpendicular_bottom_y_intercept

		'''
		y_top_left_corner = (perpendicular_slope*(xMax-5)) + perpendicular_top_y_intercept
		y_top_right_corner = (perpendicular_slope*(xMax+5)) + perpendicular_top_y_intercept
		y_bottom_left_corner = (perpendicular_slope*(xMin-5)) + perpendicular_bottom_y_intercept
		y_bottom_right_corner = (perpendicular_slope * (xMin + 5)) + perpendicular_bottom_y_intercept
		'''

	else:
		x1 = (yMin-endpoint_line_y_intercept)/endpoint_line_slope
		x2 = (yMax-endpoint_line_y_intercept)/endpoint_line_slope
		perpendicular_top_y_intercept = yMax - (perpendicular_slope * x2)
		perpendicular_bottom_y_intercept = yMin - (perpendicular_slope * x1)

		x = Symbol('x')
		bottom_result = solve(((x - x1) * (x - x1)) + (((perpendicular_slope * x + perpendicular_bottom_y_intercept) - yMin) * ((perpendicular_slope * x + perpendicular_bottom_y_intercept) - yMin)) - (5 * 5), x)
		top_result = solve(((x - x2) * (x - x2)) + (((perpendicular_slope * x + perpendicular_top_y_intercept) - yMax) * ((perpendicular_slope * x + perpendicular_top_y_intercept) - yMax)) - (5 * 5), x)
		print(bottom_result)
		top_negative_x = top_result[0].evalf()
		top_positive_x = top_result[1].evalf()
		bottom_negative_x = bottom_result[0].evalf()
		bottom_positive_x = bottom_result[1].evalf()

		top_negative_y = (top_negative_x * perpendicular_slope) + perpendicular_top_y_intercept
		top_positive_y = (top_positive_x * perpendicular_slope) + perpendicular_top_y_intercept
		bottom_negative_y = (bottom_negative_x * perpendicular_slope) + perpendicular_bottom_y_intercept
		bottom_positive_y = (bottom_positive_x * perpendicular_slope) + perpendicular_bottom_y_intercept

		'''
		y_top_left_corner = (perpendicular_slope * (x2 - 5)) + perpendicular_top_y_intercept
		y_top_right_corner = (perpendicular_slope * (x2 + 5)) + perpendicular_top_y_intercept
		y_bottom_left_corner = (perpendicular_slope * (x1 - 5)) + perpendicular_bottom_y_intercept
		y_bottom_right_corner = (perpendicular_slope * (x1 + 5)) + perpendicular_bottom_y_intercept
		'''

	corners = np.array([
		[[int(top_negative_x), int(top_negative_y)]],
		[[int(top_positive_x), int(top_positive_y)]],
		[[int(bottom_negative_x), int(bottom_negative_y)]],
		[[int(bottom_positive_x), int(bottom_positive_y)]]
	])

	rect = cv2.minAreaRect(corners)
	box = cv2.boxPoints(rect)
	box = np.intp(box)
	# print("bounding box: {}".format(box))
	cv2.drawContours(input, [box], 0, (255, 0, 255), 1)
	#width = int(rect[1][0])
	#height = int(rect[1][1])
	#src_pts = box.astype("float32")
	#dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
	#M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	#warped = cv2.warpPerspective(image, M, (width, height))

	return image


def giantDistanceAwayEquation(x1, y1, slope, y_intercept, dist):
	x = Symbol('x')
	result = solve(((x-x1)*(x-x1)) + (((slope*x + y_intercept)-y1)*((slope*x + y_intercept)-y1)) - (dist*dist), x)
	negative_x2 = result[0].evalf()
	positive_x2 = result[1].evalf()

	negative_y2 = (negative_x2 * slope) + y_intercept
	positive_y2 = (positive_x2 * slope) + y_intercept
	return negative_x2, negative_y2, positive_x2, positive_y2

negative_x2, negative_y2, positive_x2, positive_y2 = giantDistanceAwayEquation(-5, 2, -2, -10,3)

#print(negative_x2, negative_y2, positive_x2, positive_y2)

input = cv2.imread('./test/alive/alive 3.png')
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
skeleton_image = zhangSuen(input).astype("float32")
normalized_skeleton_image = skeleton_image / 255.0
endpoints = findEndpoints(skeleton_image)
intput = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
if len(endpoints)==2:
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)
		result = findRectCoords(input, endpoints, xMin, xMax, yMin, yMax)

plt.figure()
plt.imshow(result)
plt.show()

'''
data = []
i=79
j=1
for file in os.listdir('./test/aliverest/'):
	if i==24:
		i+=1
	image = './test/aliverest/' + file
	input = cv2.imread(image)
	input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

	skeleton_image = zhangSuen(input).astype("float32")
	normalized_skeleton_image = skeleton_image / 255.0

	endpoints = findEndpoints(skeleton_image)
	if len(endpoints)==2:
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)

		warped = findRectCoords(input, endpoints, xMin, xMax, yMin, yMax)
		#area_between = findAreaBetweenLineAndWorm(normalized_skeleton_image, line_image, xMin, xMax, yMin, yMax)

		#data.append([area_between, 1])
		directory = './rectangles/alive/aliverect ' + str(i) + '.png'
		cv2.imwrite(directory, warped)
		print(i, ' alive picture added')
		i+=1
	else:
		continue

for file in os.listdir('./test/dead/'):
	image = './test/dead/' + file
	input = cv2.imread(image)
	input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

	skeleton_image = zhangSuen(input).astype("float32")
	normalized_skeleton_image = skeleton_image / 255.0

	endpoints = findEndpoints(skeleton_image)
	if len(endpoints)==2:
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)

		warped = findRectCoords(input, endpoints, xMin, xMax, yMin, yMax)
		# area_between = findAreaBetweenLineAndWorm(normalized_skeleton_image, line_image, xMin, xMax, yMin, yMax)

		# data.append([area_between, 1])
		directory = './rectangles/dead/deadrect ' + str(j) + '.png'
		cv2.imwrite(directory, warped)
		print(i, ' dead picture added')
		j += 1
	else:
		continue

'''
