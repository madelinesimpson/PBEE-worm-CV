import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from tensorflow.keras.preprocessing import image

file = '/Users/madelinesimpson/PycharmProjects/HTR/testworm.jpg'
img = image.load_img(file)
img_array = image.img_to_array(img, dtype="uint8")
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

with open('/Users/madelinesimpson/PycharmProjects/HTR/masks.pkl', 'rb') as fp:
    masks = pickle.load(fp)

print(masks[0])
print(masks[0]['segmentation'].shape)
print(masks[0]['segmentation'])
print(type(masks[0]['segmentation']))

'''
Show masks/segments on the image plot in different colors
'''
def show_masks(masks):
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

filtered_masks = []

half_of_image = (masks[0]['segmentation'].shape[0]*masks[0]['segmentation'].shape[1])//2

for i in range(len(masks)):
    area = np.count_nonzero(masks[i]['segmentation'])
    if (area>300 and area<half_of_image):
        filtered_masks.append(masks[i])

print("filtered: ")
print(len(filtered_masks))

'''
Algorithm to count intersecting segments as one segment
'''

combined_masks = 0
already_combined_indexes = []
width = filtered_masks[0]['segmentation'].shape[0]
height = filtered_masks[0]['segmentation'].shape[1]

for i in range(len(filtered_masks)):
    combined = False
    current_segment = filtered_masks[i]['segmentation']
    for j in range(len(filtered_masks)):
        if [j,i] in already_combined_indexes:
            break
        pixels_touching = 0
        compare_segment = filtered_masks[j]['segmentation']
        combined_segments = np.logical_or(current_segment,compare_segment)
        for w in range(width):
            for h in range(height):
                differences = 0
                if current_segment[w][h]==True:
                    if w!=0:
                        if combined_segments[w-1][h]!=current_segment[w-1][h]:
                            differences+=1
                    if h!=0:
                        if combined_segments[w][h-1]!=current_segment[w][h-1]:
                            differences+=1
                    if w!=width-1:
                        if combined_segments[w+1][h]!=current_segment[w+1][h]:
                            differences += 1
                    if h!=height-1:
                        if combined_segments[w][h+1]!=current_segment[w][h+1]:
                            differences += 1
                    if differences>0:
                        pixels_touching+=1
        if pixels_touching>10:
            combined_masks+=1
            already_combined_indexes.append([i,j])

num_worms = len(filtered_masks)-combined_masks

print("number of worms:")
print(num_worms)

'''
Show the plot
'''

plt.figure(figsize=(10, 10))
plt.imshow(img_array)
show_masks(filtered_masks)
plt.axis('off')
plt.show()
