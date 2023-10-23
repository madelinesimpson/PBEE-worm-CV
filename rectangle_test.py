import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sympy.solvers import solve
from sympy import Symbol


#-----------------------------------CODE TAKEN FROM THIEN WORMS GITHUB-----------------------------------

#Get the neighbors of a particular pixel in an image
def neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

#The same thing? I took this portion of the code so not sure why they repeat function
def get_neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
				img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

#Helper function for the skeleton algorithm
def transitions(neighbour, white):
	count = 0
	n = neighbour + neighbour[0:1]
	return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )

#Algorithm to turn a worm into its skeleton
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

#Find the endpoints of the skeleton of the worm
def findEndpoints(skeleton):
	(rows,cols) = np.nonzero(skeleton)
	skel_coords = []

	for (r,c) in zip(rows,cols):
		counter = countNeighbouringPixels(skeleton, r,c)
		if counter == 1:
			skel_coords.append((r,c))
	return skel_coords

#Counts how many white pixels surround a particular pixel
def countNeighbouringPixels(skeleton, x, y):
    neighbours = get_neighbours(x,y,skeleton)
    return sum(neighbours) / 255

#------------------------------------------------------------------------------------


#-----------------------------------MY CODE-----------------------------------

#Takes a normalized version of the skeleton image (1 = white, 0 = black) and returns the Min and Max X and Y values of the worm in the image
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

#Takes the worm and crops it into a rectangle that is created along the line drawn between the two endpoints of the worm
def rectangleCrop(image, endpoints, xMin, xMax, yMin, yMax):
	xDist = xMax-xMin
	yDist = yMax-yMin
	endpoint_x1 = endpoints[0][1]
	endpoint_x2 = endpoints[1][1]
	endpoint_y1 = endpoints[0][0]
	endpoint_y2 = endpoints[1][0]

	vert = False

	#Find equation of line between endpoints
	if endpoint_x1==endpoint_x2:
		vert = True
		endpoint_line_slope=100000
	else:
		endpoint_line_slope = (endpoint_y2-endpoint_y1)/(endpoint_x2-endpoint_x1)
		endpoint_line_y_intercept = endpoint_y1 - (endpoint_line_slope*endpoint_x1)

	#Make sure we dont divide by zero and calculate the slope of the line perpendicular to the line between endpoints to use as rectangle boundaries
	if endpoint_line_slope==0 or vert==True:
		perpendicular_slope=0
	else:
		perpendicular_slope = -(1 / endpoint_line_slope)

	#Array we will put the coordinates of the rectangle corners in
	corners = []

	#Special case for if the slope of the endpoint line is 0
	if vert==True:
		if yDist <= 100:
			rect_thick = 4
		elif yDist > 100 and yDist < 300:
			rect_thick = 6
		else:
			rect_thick = 8

		top_negative_x = xMax
		top_negative_y = yMax - rect_thick
		top_positive_x = xMax
		top_positive_y = yMax + rect_thick
		bottom_negative_x = xMin
		bottom_negative_y = yMin - rect_thick
		bottom_positive_x = xMin
		bottom_positive_y = yMin + rect_thick

	elif endpoint_line_slope==0:
		#Use length of line extended to the min and max X of the worm to determine how thick rectangle will be
		if xDist <= 100:
			rect_thick = 4
		elif xDist > 100 and xDist < 300:
			rect_thick = 6
		else:
			rect_thick = 8

		#Set the bounding x and y coordinates of the rectangle corners
		top_negative_x = xMax
		top_negative_y = xMax - rect_thick
		top_positive_x = xMax
		top_positive_y = xMax + rect_thick
		bottom_negative_x = xMin
		bottom_negative_y = xMin - rect_thick
		bottom_positive_x = xMin
		bottom_positive_y = xMin + rect_thick

	#If the worm is horizontal, solve for rectangle like this:,
	elif ((abs(endpoint_line_slope)<1) or (abs(endpoint_line_slope==1) and xDist>yDist)):
		#Extending endpoint line to the xMin and xMax boundaries of the whole worm
		y_for_x_min = xMin * endpoint_line_slope + endpoint_line_y_intercept
		y_for_x_max = xMax * endpoint_line_slope + endpoint_line_y_intercept

		#Find length of new line and use to determine how thick the rectangle will be
		line_length = math.dist((xMin, y_for_x_min), (xMax, y_for_x_max))
		if line_length <= 100:
			rect_thick = 4
		elif line_length > 100 and line_length < 300:
			rect_thick = 6
		else:
			rect_thick = 8

		#Find the equation of the two lines perpendicular to the endpoint line that intersect at the xMin and xMax
		perpendicular_top_y_intercept = y_for_x_max - (perpendicular_slope*xMax)
		perpendicular_bottom_y_intercept = y_for_x_min - (perpendicular_slope*xMin)

		#Really ugly equation I derived to calculate the coordinates for each perpendicular line that extend it by rect_thick in each direction (the corners of the rectangle)
		x = Symbol('x')
		top_result = solve(((x - xMax) * (x - xMax)) + (((perpendicular_slope * x + perpendicular_top_y_intercept) - y_for_x_max) * ((perpendicular_slope * x + perpendicular_top_y_intercept) - y_for_x_max)) - (rect_thick * rect_thick), x)
		bottom_result = solve(((x - xMin) * (x - xMin)) + (((perpendicular_slope * x + perpendicular_bottom_y_intercept) - y_for_x_min) * ((perpendicular_slope * x + perpendicular_bottom_y_intercept) - y_for_x_min)) - (rect_thick * rect_thick), x)
		#Get the x values of the 4 points
		top_negative_x = top_result[0].evalf()
		top_positive_x = top_result[1].evalf()
		bottom_negative_x = bottom_result[0].evalf()
		bottom_positive_x = bottom_result[1].evalf()

		#Get the y values of the 4 points
		top_negative_y = (top_negative_x * perpendicular_slope) + perpendicular_top_y_intercept
		top_positive_y = (top_positive_x * perpendicular_slope) + perpendicular_top_y_intercept
		bottom_negative_y = (bottom_negative_x * perpendicular_slope) + perpendicular_bottom_y_intercept
		bottom_positive_y = (bottom_positive_x * perpendicular_slope) + perpendicular_bottom_y_intercept

	#If worm is vertical, solve for rectangle like this:
	else:
		#Extending endpoint line to the yMin and yMax boundaries of the whole worm
		x1 = (yMin-endpoint_line_y_intercept)/endpoint_line_slope
		x2 = (yMax-endpoint_line_y_intercept)/endpoint_line_slope

		# Find length of new line and use to determine how thick the rectangle will be
		line_length = math.dist((x1, yMin), (x2, yMax))
		if line_length <= 100:
			rect_thick = 4
		elif line_length > 100 and line_length < 300:
			rect_thick = 6
		else:
			rect_thick = 8

		#Find the equation of the two lines perpendicular to the endpoint line that intersect at the yMin and yMax
		perpendicular_top_y_intercept = yMax - (perpendicular_slope * x2)
		perpendicular_bottom_y_intercept = yMin - (perpendicular_slope * x1)

		# Really ugly equation I derived to calculate the coordinates for each perpendicular line that extend it by rect_thick in each direction (the corners of the rectangle)
		x = Symbol('x')
		bottom_result = solve(((x - x1) * (x - x1)) + (((perpendicular_slope * x + perpendicular_bottom_y_intercept) - yMin) * ((perpendicular_slope * x + perpendicular_bottom_y_intercept) - yMin)) - (rect_thick * rect_thick), x)
		top_result = solve(((x - x2) * (x - x2)) + (((perpendicular_slope * x + perpendicular_top_y_intercept) - yMax) * ((perpendicular_slope * x + perpendicular_top_y_intercept) - yMax)) - (rect_thick * rect_thick), x)

		# Get the x values of the 4 points
		top_negative_x = top_result[0].evalf()
		top_positive_x = top_result[1].evalf()
		bottom_negative_x = bottom_result[0].evalf()
		bottom_positive_x = bottom_result[1].evalf()

		# Get the y values of the 4 points
		top_negative_y = (top_negative_x * perpendicular_slope) + perpendicular_top_y_intercept
		top_positive_y = (top_positive_x * perpendicular_slope) + perpendicular_top_y_intercept
		bottom_negative_y = (bottom_negative_x * perpendicular_slope) + perpendicular_bottom_y_intercept
		bottom_positive_y = (bottom_positive_x * perpendicular_slope) + perpendicular_bottom_y_intercept

	#Put the x and y values of the found rectangle corners into an array
	corners = np.array([
		[[int(top_negative_x), int(top_negative_y)]],
		[[int(top_positive_x), int(top_positive_y)]],
		[[int(bottom_negative_x), int(bottom_negative_y)]],
		[[int(bottom_positive_x), int(bottom_positive_y)]]
	])

	#Find the rectangle between the 4 corners
	rect = cv2.minAreaRect(corners)
	box = cv2.boxPoints(rect)
	box = np.intp(box)

	#Find width and height of rectangle
	width = int(rect[1][0])
	height = int(rect[1][1])
	src_pts = box.astype("float32")

	#Find the matrix that warps/crops the image to the rectangle idk I stole this snippet from online and it works
	dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	warped = cv2.warpPerspective(image, M, (width, height))

	#Return the rectangle image (it has to be warped bc the rectangles are mostly on a diagonal axis)
	return warped

#Testing code I was using for one image

'''
input = cv2.imread('./test/alive/alive 9.png')
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
skeleton_image = zhangSuen(input).astype("float32")
normalized_skeleton_image = skeleton_image / 255.0
endpoints = findEndpoints(skeleton_image)
print(endpoints)
input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
if len(endpoints)==2:
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)
		print(xMin, xMax, yMin, yMax)
		result = rectangleCrop(input, endpoints, xMin, xMax, yMin, yMax)

plt.figure()
plt.imshow(input)
plt.show()
'''

#Applying rectangle transformation to all the worms in the dataset
alive_list=[]
dead_list=[]
for file in os.listdir('./test/alive/'):
	alive_list.append(file)

for file in os.listdir('./test/dead/'):
	dead_list.append(file)

alive_list.sort(key=lambda test_string: list(
		map(int, re.findall(r'\d+', test_string)))[0])

dead_list.sort(key=lambda test_string: list(
		map(int, re.findall(r'\d+', test_string)))[0])

i=1
j=1

data = []
for file in alive_list:
	image = './test/alive/' + file
	input = cv2.imread(image)
	input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

	#Find the skeleton of the worm
	skeleton_image = zhangSuen(input).astype("float32")
	#Normalize it
	normalized_skeleton_image = skeleton_image / 255.0
	#Find the endpoints of the worm
	endpoints = findEndpoints(skeleton_image)
	#If the worm has two endpoints (it is not a cluster a worm curled up in a circle) proceed
	if len(endpoints)==2:
		#Find the bounding coordinates of the worm
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)
		#Get the rectangle iamge
		warped = rectangleCrop(input, endpoints, xMin, xMax, yMin, yMax)
		#Rotate to horizontal if image is vertical
		if(warped.shape[0]>warped.shape[1]):
			 warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
		#Write it to a new directory
		directory = './rectangles/alive/aliverect ' + str(i) + '.png'
		cv2.imwrite(directory, warped)
		print(file, 'made into rectangle')
		i+=1
	else:
		print(file + " skipped")
		i+=1
		continue

#Do the same for the directory of dead worms
for file in dead_list:
	image = './test/dead/' + file
	input = cv2.imread(image)
	input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
	skeleton_image = zhangSuen(input).astype("float32")
	normalized_skeleton_image = skeleton_image / 255.0
	endpoints = findEndpoints(skeleton_image)
	if len(endpoints)==2:
		xMin,xMax,yMin,yMax = findBoundingCoords(normalized_skeleton_image)
		warped = rectangleCrop(input, endpoints, xMin, xMax, yMin, yMax)
		if(warped.shape[0]>warped.shape[1]):
			 warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
		directory = './rectangles/dead/deadrect ' + str(j) + '.png'
		cv2.imwrite(directory, warped)
		print(file, 'made into rectangle')
		j += 1
	else:
		print(file + " skipped")
		j += 1
		continue
