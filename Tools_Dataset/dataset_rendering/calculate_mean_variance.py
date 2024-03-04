import cv2
import numpy as np
import os
import glob
import shutil

renders_path = "renders"

# read all folders in renders_path
room_paths = glob.glob(os.path.join(renders_path, "*"))

print(f"Found {len(room_paths)} rooms")

# float64 total_sum
total_sum = 0

total_num_of_pixels = 0

for i in range(len(room_paths)):

    room_path = room_paths[i]

    ldi_path = os.path.join(room_path, "ldi.npy")
    rgb_path = os.path.join(room_path, "rgb.png")

    if not os.path.exists(ldi_path) or not os.path.exists(rgb_path):
        # delete folder with its files shutil
        shutil.rmtree(room_path)
        continue

    ldi = np.load(ldi_path)

    # flatten ldi
    ldi = ldi.flatten()

    # remove zeros from ldi
    ldi = ldi[ldi >= 0.01]
    # ldi = ldi[ldi != 0]

    # sum of ldi
    room_sum = np.sum(ldi)

    # add to total sum
    total_sum += room_sum

    # add to total number of pixels
    total_num_of_pixels += len(ldi)

    # print(f"Processed {i+1}/{len(room_paths)}")


mean = total_sum / total_num_of_pixels

print(f"Mean: {mean}")

# Now calculate the standard deviation
sum_squared_diff = 0

for i in range(len(room_paths)):
    
    room_path = room_paths[i]

    ldi_path = os.path.join(room_path, "ldi.npy")
    rgb_path = os.path.join(room_path, "rgb.png")

    if not os.path.exists(ldi_path) or not os.path.exists(rgb_path):
        # delete folder with its files shutil
        continue

    ldi = np.load(ldi_path)

    # flatten ldi
    ldi = ldi.flatten()

    # remove zeros from ldi
    ldi = ldi[ldi >= 0.01]

    # sum of squared differences from the mean
    sum_squared_diff += np.sum((ldi - mean) ** 2)

std_dev = np.sqrt(sum_squared_diff / total_num_of_pixels)

print(f"Standard Deviation: {std_dev}")

# old_renders2
# Mean: 2.5502266793814417
# Standard Deviation: 0.9683916122629385

# renders
# Found 7708 rooms
# Mean: 2.450134528114036
# Standard Deviation: 1.1005151495167442



