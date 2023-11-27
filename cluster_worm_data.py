import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import individual_worm_data as worm

def get_worm_neighbour_indices(x,y,skeleton):
    neighbours = []
    indices = []
    if skeleton[x-1][y]==1:
        neighbours.append([x-1,y])
        indices.append(1)
    if skeleton[x-1][y+1] == 1:
        neighbours.append([x-1, y+1])
        indices.append(2)
    if skeleton[x][y+1] == 1:
        neighbours.append([x, y+1])
        indices.append(3)
    if skeleton[x+1][y+1] == 1:
        neighbours.append([x+1, y+1])
        indices.append(4)
    if skeleton[x+1][y] == 1:
        neighbours.append([x+1, y])
        indices.append(5)
    if skeleton[x+1][y-1] == 1:
        neighbours.append([x+1, y-1])
        indices.append(6)
    if skeleton[x][y-1] == 1:
        neighbours.append([x, y-1])
        indices.append(7)
    if skeleton[x-1][y-1] == 1:
        neighbours.append([x-1, y-1])
        indices.append(8)
    return neighbours, indices

def get_perp_directions(direction):
    perp_directions = []
    if direction==0:
        perp_directions = [3,4,6,7]
    elif direction==1:
        perp_directions = [4,5,7,8]
    elif direction==2:
        perp_directions = [1,5,6,8]
    elif direction==3:
        perp_directions = [1,2,6,7]
    elif direction==4:
        perp_directions = [2,3,7,8]
    elif direction==5:
        perp_directions = [1,3,4,8]
    elif direction==6:
        perp_directions = [1,2,4,5]
    elif direction==7:
        perp_directions = [2,3,5,6]
    else:
        perp_directions=[]
    return perp_directions


def find_intersections(skeleton, endpoints):
    skeleton = skeleton/255.0
    intersections = []

    for coord in endpoints:
        direction = -1
        path = []
        x = coord[0]
        y = coord[1]
        n = worm.get_neighbours(x, y, skeleton)
        directions = [[x - 1, y], [x - 1, y + 1], [x, y + 1], [x + 1, y + 1], [x + 1, y], [x + 1, y - 1], [x, y - 1],
                      [x - 1, y - 1]]
        for i in range(0, len(n)):
            if n[i]==1:
                direction=i
        perp = False
        perp_directions = get_perp_directions(direction)
        path.append((x,y))
        prev_x = x
        prev_y = y
        prev_prev_x = -1
        prev_prev_y = -1
        current_x = directions[direction][0]
        current_y = directions[direction][1]
        while perp==False:
            path.append([current_x, current_y])
            print(current_x, ", ", current_y)
            neighbours, indices = get_worm_neighbour_indices(current_x, current_y, skeleton)
            n = worm.get_neighbours(current_x, current_y, skeleton)
            pixel_count = np.count_nonzero(n)
            if pixel_count>4:
                print("intersection found > 4")
                intersections.append([current_x, current_y])
                if len(path)<10:
                    for point in path:
                        skeleton[point[0]][point[1]] = 0
                perp=True
            else:
                prev_point = [prev_x, prev_y]
                prev_prev_point = [prev_prev_x, prev_prev_y]
                print(len(indices))
                if prev_prev_point in neighbours:
                    index = neighbours.index(prev_prev_point)
                    neighbours.remove(prev_prev_point)
                    indices.pop(index)
                if prev_point in neighbours:
                    index = neighbours.index(prev_point)
                    neighbours.remove(prev_point)
                    indices.pop(index)
                if len(neighbours)==2:
                    direction_of_point_one = indices[0]
                    direction_of_point_two = indices[1]
                    if direction_of_point_one in perp_directions:
                        point = neighbours[0]
                        new_neighbours, indices = get_worm_neighbour_indices(point[0], point[1], skeleton)
                        if direction_of_point_one in indices:
                            print("perp intersect found")
                            intersections.append([current_x, current_y])
                            perp=True
                        else:
                            prev_prev_x = prev_x
                            prev_prev_y = prev_y
                            prev_x = current_x
                            prev_y = current_y
                            if direction_of_point_one == 1 or direction_of_point_one == 3 or direction_of_point_one == 5 or direction_of_point_one == 7:
                                current_x = neighbours[0][0]
                                current_y = neighbours[0][1]
                            else:
                                current_x = neighbours[1][0]
                                current_y = neighbours[1][1]
                    elif direction_of_point_two in perp_directions:
                        point = neighbours[1]
                        new_neighbours, indices = get_worm_neighbour_indices(point[0], point[1], skeleton)
                        if direction_of_point_two in indices:
                            print("perp intersect found")
                            intersections.append([current_x, current_y])
                            perp=True
                        else:
                            prev_prev_x = prev_x
                            prev_prev_y = prev_y
                            prev_x = current_x
                            prev_y = current_y
                            if direction_of_point_one == 1 or direction_of_point_one == 3 or direction_of_point_one == 5 or direction_of_point_one == 7:
                                current_x = neighbours[0][0]
                                current_y = neighbours[0][1]
                            else:
                                current_x = neighbours[1][0]
                                current_y = neighbours[1][1]
                    else:
                        prev_prev_x = prev_x
                        prev_prev_y = prev_y
                        prev_x = current_x
                        prev_y = current_y
                        if direction_of_point_one == 1 or direction_of_point_one == 3 or direction_of_point_one == 5 or direction_of_point_one == 7:
                            current_x = neighbours[0][0]
                            current_y = neighbours[0][1]
                        else:
                            current_x = neighbours[1][0]
                            current_y = neighbours[1][1]
                else:
                    print(len(indices))
                    direction_of_point = indices[0]
                    if direction_of_point in perp_directions:
                        point = neighbours[0]
                        new_neighbours, indices = get_worm_neighbour_indices(point[0], point[1], skeleton)
                        if direction_of_point in indices:
                            print("perp intersect found")
                            intersections.append([current_x, current_y])
                            perp = True
                        else:
                            prev_prev_x = prev_x
                            prev_prev_y = prev_y
                            prev_x = current_x
                            prev_y = current_y
                            current_x = neighbours[0][0]
                            current_y = neighbours[0][1]
                    else:
                        prev_prev_x = prev_x
                        prev_prev_y = prev_y
                        prev_x = current_x
                        prev_y = current_y
                        current_x = neighbours[0][0]
                        current_y = neighbours[0][1]
        print("endpoint")
    skeleton = skeleton * 255.0
    return intersections, skeleton

def find_endpoints(skeleton):
    skeleton = skeleton/255.0
    (rows,cols) = np.nonzero(skeleton)
    skel_coords = []
    for (r,c) in zip(rows,cols):
        n = worm.get_neighbours(r, c, skeleton)
        num_pixels = np.count_nonzero(n)
        if num_pixels == 1:
            skel_coords.append((r,c))
    return skel_coords


'''
def find_intersections(skeleton, xMin, xMax, yMin, yMax):
    intersection_points = []
    skeleton = skeleton / 255.0
    for x in range(yMin + 1, yMax - 1):
        for y in range(xMin + 1, xMax - 1):
            n = worm.get_neighbours(x, y, skeleton)
            n.append(skeleton[x][y])
            num_pixels = np.count_nonzero(n)
            if num_pixels>5 and len(intersection_points)==0:
                intersection_points.append((x, y))
            elif num_pixels>5 and (abs(intersection_points[-1][0]-x)>3 and abs(intersection_points[-1][1]-y)>3):
                intersection_points.append((x,y))
            else:
                continue
    return intersection_points
'''

#if len(path) < 10:
                            #for point in path:
                                #skeleton[point[0]][point[1]] = 0
