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
    direction = direction + 1
    if direction==1:
        perp_directions = [3,4,6,7]
    elif direction==2:
        perp_directions = [4,5,7,8]
    elif direction==3:
        perp_directions = [1,5,6,8]
    elif direction==4:
        perp_directions = [1,2,6,7]
    elif direction==5:
        perp_directions = [2,3,7,8]
    elif direction==6:
        perp_directions = [1,3,4,8]
    elif direction==7:
        perp_directions = [1,2,4,5]
    elif direction==8:
        perp_directions = [2,3,5,6]
    else:
        perp_directions=[]
    return perp_directions

def get_direction(x, y, skeleton):
    direction = -1
    n = worm.get_neighbours(x, y, skeleton)
    for i in range(0, len(n)):
        if n[i] == 1:
            direction = i
    return direction

def move_forward_one(current_x, current_y, prev_x, prev_y, neighbours):
    prev_prev_x = prev_x
    prev_prev_y = prev_y
    prev_x = current_x
    prev_y = current_y
    current_x = neighbours[0][0]
    current_y = neighbours[0][1]
    return current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y

def move_forward_two(current_x, current_y, prev_x, prev_y, direction_of_point_one, direction_of_point_two, neighbours):
    prev_prev_x = prev_x
    prev_prev_y = prev_y
    prev_x = current_x
    prev_y = current_y
    if direction_of_point_one == 1 or direction_of_point_one == 3 or direction_of_point_one == 5 or direction_of_point_one == 7:
        current_x = neighbours[0][0]
        current_y = neighbours[0][1]
    elif direction_of_point_two == 1 or direction_of_point_two == 3 or direction_of_point_two == 5 or direction_of_point_two == 7:
        current_x = neighbours[1][0]
        current_y = neighbours[1][1]
    else:
        current_x = neighbours[0][0]
        current_y = neighbours[0][1]
    return current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y

def move_intersection(x, y, skeleton):
    neighbours, indices = get_worm_neighbour_indices(x, y, skeleton)
    if len(indices)==0:
        return -1000
    if 1 in indices:
        return [x-1,y]
    elif 3 in indices:
        return [x, y+1]
    elif 5 in indices:
        return [x+1, y-1]
    elif 7 in indices:
        return [x-1, y-1]
    elif 2 in indices:
        return [x-1, y+1]
    elif 4 in indices:
        return [x+1, y+1]
    elif 6 in indices:
        return [x+1, y-1]
    else:
        return [x-1, y-1]


def find_intersections(skeleton, endpoints):
    skeleton = skeleton/255.0
    intersections = []
    removed_endpoints = []
    new_endpoints = []

    for coord in endpoints:
        path = []
        perp = False
        x, y = coord[0], coord[1]
        path.append((x, y))
        directions = [[x - 1, y], [x - 1, y + 1], [x, y + 1], [x + 1, y + 1], [x + 1, y], [x + 1, y - 1], [x, y - 1],
                      [x - 1, y - 1]]
        direction = get_direction(x,y,skeleton)
        perp_directions = get_perp_directions(direction)
        current_x, current_y = directions[direction][0], directions[direction][1]
        prev_x, prev_y = x, y
        prev_prev_x, prev_prev_y = -1, -1

        while perp==False:

            path.append([current_x, current_y])
            neighbours, indices = get_worm_neighbour_indices(current_x, current_y, skeleton)
            n = worm.get_neighbours(current_x, current_y, skeleton)
            pixel_count = np.count_nonzero(n)

            if pixel_count>4:
                #print("intersection found > 4")
                if len(path) < 15:
                    for point in path:
                        skeleton[point[0]][point[1]] = 0
                    removed_endpoints.append(coord)
                    new_endpoints.append([current_x, current_y])
                else:
                    intersections.append([current_x, current_y])
                perp=True

            else:
                prev_point = [prev_x, prev_y]
                prev_prev_point = [prev_prev_x, prev_prev_y]

                if prev_prev_point in neighbours:
                    index = neighbours.index(prev_prev_point)
                    neighbours.remove(prev_prev_point)
                    indices.pop(index)

                if prev_point in neighbours:
                    index = neighbours.index(prev_point)
                    neighbours.remove(prev_point)
                    indices.pop(index)

                if len(neighbours)==2:
                    direction_of_point_one, direction_of_point_two = indices[0], indices[1]

                    if direction_of_point_one in perp_directions:
                        point = neighbours[0]
                        new_neighbours, new_indices = get_worm_neighbour_indices(point[0], point[1], skeleton)

                        if direction_of_point_one in new_indices:
                            #print("intersection found")
                            if len(path) < 15:
                                for point in path:
                                    skeleton[point[0]][point[1]] = 0
                                removed_endpoints.append(coord)
                                new_endpoints.append([current_x, current_y])
                            else:
                                intersections.append([current_x, current_y])
                            perp=True

                        else:
                            current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_two(current_x, current_y, prev_x, prev_y, direction_of_point_one, direction_of_point_two, neighbours)

                    elif direction_of_point_two in perp_directions:
                        point = neighbours[1]
                        new_neighbours, new_indices = get_worm_neighbour_indices(point[0], point[1], skeleton)

                        if direction_of_point_two in new_indices:
                            #print("intersection found")
                            if len(path) < 15:
                                for point in path:
                                    skeleton[point[0]][point[1]] = 0
                                removed_endpoints.append(coord)
                                new_endpoints.append([current_x, current_y])
                            else:
                                intersections.append([current_x, current_y])
                            perp=True

                        else:
                            current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_two(current_x, current_y, prev_x, prev_y, direction_of_point_one, direction_of_point_two, neighbours)

                    else:
                        current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_two(current_x, current_y, prev_x, prev_y, direction_of_point_one, direction_of_point_two, neighbours)

                else:
                    direction_of_point = indices[0]

                    if direction_of_point in perp_directions:
                        point = neighbours[0]
                        new_neighbours, new_indices = get_worm_neighbour_indices(point[0], point[1], skeleton)
                        if direction_of_point in new_indices:
                            #print("intersection found")
                            if len(path) < 15:
                                for point in path:
                                    skeleton[point[0]][point[1]] = 0
                                removed_endpoints.append(coord)
                                new_endpoints.append([current_x, current_y])
                            else:
                                intersections.append([current_x, current_y])
                            perp = True

                        else:
                            current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_one(current_x, current_y, prev_x, prev_y, neighbours)

                    else:
                        current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_one(current_x, current_y, prev_x, prev_y, neighbours)
        #print("endpoint")

    skeleton = skeleton * 255.0
    return intersections, removed_endpoints, new_endpoints, skeleton

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def remove_repeat_intersections(points, d, skeleton):
    skeleton = skeleton / 255.0
    intersections = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] = point[0] // count
            point[1] = point[1] // count
            intersections.append([point[0], point[1]])
    for coord in intersections:
        if skeleton[coord[0]][coord[1]]==1:
            continue
        else:
            new_coord = move_intersection(coord[0], coord[1], skeleton)
            if new_coord == -1000:
                print('uh oh')
            else:
                intersections.remove(coord)
                intersections.append(new_coord)
    return intersections

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

def get_intersection_sphere(x,y):
    neighbours = [[x - 1, y], [x - 1, y + 1], [x, y + 1], [x + 1, y + 1], [x + 1, y], [x + 1, y - 1], [x, y - 1], [x - 1, y - 1]]
    return neighbours


def get_branches(intersections, endpoints, skeleton):
    skeleton = skeleton/255.0
    branches = []
    for coord in endpoints:
        path = []
        x, y = coord[0], coord[1]
        path.append([x, y])
        init_neighbours, init_indices = get_worm_neighbour_indices(x,y,skeleton)
        current_x, current_y = init_neighbours[0][0], init_neighbours[0][1]
        prev_x, prev_y = x, y
        prev_prev_x, prev_prev_y = -1, -1
        touching_intersection = False
        skip = False
        while touching_intersection==False:
            path.append([current_x, current_y])
            current_point = [current_x, current_y]
            for point in intersections:
                sphere = get_intersection_sphere(point[0], point[1])
                #print(sphere)
                if current_point in sphere:
                    touching_intersection=True
                    skip=True
                    if len(path) < 15:
                        continue
                    else:
                        endpoints = []
                        endpoints.append([x,y])
                        endpoints.append(current_point)
                        temp_dict = {
                            "path": path,
                            "endpoints": endpoints,
                        }
                        branches.append(temp_dict)
                        #print('branch found')
            if skip==True:
                continue
            else:
                neighbours, indices = get_worm_neighbour_indices(current_x, current_y, skeleton)
                prev_point = [prev_x, prev_y]
                prev_prev_point = [prev_prev_x, prev_prev_y]
                if prev_prev_point in neighbours:
                    index = neighbours.index(prev_prev_point)
                    neighbours.remove(prev_prev_point)
                    indices.pop(index)
                if prev_point in neighbours:
                    index = neighbours.index(prev_point)
                    neighbours.remove(prev_point)
                    indices.pop(index)
                if len(neighbours)==2:
                    direction_of_point_one, direction_of_point_two = indices[0], indices[1]
                    current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_two(current_x, current_y, prev_x, prev_y, direction_of_point_one, direction_of_point_two, neighbours)
                else:
                    current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_one(current_x, current_y, prev_x, prev_y, neighbours)

    undone_points = []
    for coord in intersections:
        int_neighbours, int_indices = get_worm_neighbour_indices(coord[0], coord[1], skeleton)
        for branch in branches:
            path = branch['path']
            for n in int_neighbours:
                if n in path:
                    index = int_neighbours.index(n)
                    int_neighbours.remove(n)
                    int_indices.pop(index)
            for left in int_neighbours:
                undone_points.append(left)

    final_undone = []
    for p in undone_points:
        if p in final_undone:
            continue
        else:
            final_undone.append(p)

    return branches, final_undone


def get_paths_between_intersections(intersections, undone_points, skeleton):
    skeleton = skeleton / 255.0
    branches = []
    for coord in undone_points:
        x, y = coord[0], coord[1]
        init_neighbours, init_indices = get_worm_neighbour_indices(x, y, skeleton)
        for point in init_neighbours:
            if point in intersections: #or point in init_sphere:
                index = init_neighbours.index(point)
                init_neighbours.remove(point)
                init_indices.pop(index)
        if len(init_neighbours) > 1:
            for i in range(0, len(init_neighbours)):
                path = []
                path.append([x, y])
                if init_indices[i]==2 and ((1 in init_indices) or (3 in init_indices)):
                    continue
                elif init_indices[i] == 4 and ((3 in init_indices) or (5 in init_indices)):
                    continue
                elif init_indices[i] == 6 and ((5 in init_indices) or (7 in init_indices)):
                    continue
                elif init_indices[i] == 8 and ((1 in init_indices) or (7 in init_indices)):
                    continue
                else:
                    current_x, current_y = init_neighbours[i][0], init_neighbours[i][1]
                    prev_x, prev_y = x, y
                    prev_prev_x, prev_prev_y = -1, -1
                    touching_intersection = False
                    skip = False
                    while touching_intersection == False:
                        path.append([current_x, current_y])
                        current_point = [current_x, current_y]
                        for intersect in intersections:
                            sphere = get_intersection_sphere(intersect[0], intersect[1])
                            if current_point in sphere:
                                touching_intersection = True
                                skip = True
                                if len(path) > 10:
                                    endpoints = []
                                    endpoints.append([x, y])
                                    endpoints.append(current_point)
                                    temp_dict = {
                                        "path": path,
                                        "endpoints": endpoints,
                                    }
                                    branches.append(temp_dict)
                                    #print('branch found')
                        if skip == True:
                            continue
                        else:
                            neighbours, indices = get_worm_neighbour_indices(current_x, current_y, skeleton)
                            prev_point = [prev_x, prev_y]
                            prev_prev_point = [prev_prev_x, prev_prev_y]
                            if prev_prev_point in neighbours:
                                index = neighbours.index(prev_prev_point)
                                neighbours.remove(prev_prev_point)
                                indices.pop(index)
                            if prev_point in neighbours:
                                index = neighbours.index(prev_point)
                                neighbours.remove(prev_point)
                                indices.pop(index)
                            if len(neighbours) == 0:
                                touching_intersection = True
                            elif len(neighbours) == 2:
                                direction_of_point_one, direction_of_point_two = indices[0], indices[1]
                                current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_two(current_x,
                                                                                                                  current_y, prev_x,
                                                                                                                  prev_y,
                                                                                                                  direction_of_point_one,
                                                                                                                  direction_of_point_two,
                                                                                                                  neighbours)
                            else:
                                current_x, current_y, prev_x, prev_y, prev_prev_x, prev_prev_y = move_forward_one(current_x,
                                                                                                                  current_y, prev_x,
                                                                                                                  prev_y,
                                                                                                               neighbours)
    return branches

def remove_repeat_branches(branches):
    unique_branches = []
    for path in branches:
        if len(unique_branches)==0:
            unique_branches.append(path)
        else:
            done=False
            for other_path in unique_branches:
                if done==True:
                    break
                else:
                    in_common = 0
                    for point in path:
                        for other_point in other_path:
                            if point==other_point:
                                in_common += 1
                    if in_common > 5:
                        done=True
                    else:
                        continue
            unique_branches.append(path)
    return unique_branches

def get_branch_og_worm(branch, image):

    return
