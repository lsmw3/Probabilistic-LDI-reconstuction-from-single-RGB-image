import os
import shutil
import glob
import random
import numpy as np

folder = "val"

# read all the folders in the main folder
folders = glob.glob(folder + "/*")

coverage_list = []
coverage_list_mask = []
coverage_list_square = []

count = 0
for folder in folders:
    ldi_path = os.path.join(folder, "ldi.npy")
    ldi = np.load(ldi_path) # (1024, 1024, 10)
    ldi_mask = np.where(ldi > 0, 1, 0)
    ldi_square = ldi * ldi

    ldi_coverage_list = []
    ldi_mask_coverage_list = []
    ldi_square_coverage_list = []

    total_ldi_sum = 0
    total_ldi_mask_sum = 0
    total_ldi_square_sum = 0

    for c in range(ldi.shape[2]):
        ldi_coverage = np.sum(ldi[:, :, c])
        ldi_mask_coverage = np.sum(ldi_mask[:, :, c])
        ldi_square_coverage = np.sum(ldi_square[:, :, c])


        total_ldi_sum += ldi_coverage
        total_ldi_mask_sum += ldi_mask_coverage
        total_ldi_square_sum += ldi_square_coverage

        
        # ldi_coverage_list.append(ldi_coverage)
        # ldi_mask_coverage_list.append(ldi_mask_coverage)
        # ldi_square_coverage_list.append(ldi_square_coverage)
        
        ldi_coverage_list.append(total_ldi_sum)
        ldi_mask_coverage_list.append(total_ldi_mask_sum)
        ldi_square_coverage_list.append(total_ldi_square_sum)

    for i in range(len(ldi_coverage_list)):
        ldi_coverage_list[i] = ldi_coverage_list[i] / total_ldi_sum
        ldi_mask_coverage_list[i] = ldi_mask_coverage_list[i] / total_ldi_mask_sum
        ldi_square_coverage_list[i] = ldi_square_coverage_list[i] / total_ldi_square_sum

    count += 1
    print("Count: ", count, " / ", len(folders))

    coverage_list.append(ldi_coverage_list)
    coverage_list_mask.append(ldi_mask_coverage_list)
    coverage_list_square.append(ldi_square_coverage_list)

coverage_list = np.array(coverage_list)
coverage_list_mask = np.array(coverage_list_mask)
coverage_list_square = np.array(coverage_list_square)

mean_coverage_list = np.mean(coverage_list, axis=0)
mean_coverage_list_mask = np.mean(coverage_list_mask, axis=0)
mean_coverage_list_square = np.mean(coverage_list_square, axis=0)


for c in range(len(mean_coverage_list)):
    print("Num LDI: ", c+1)
    print("Mean coverage: ", mean_coverage_list[c])
    print("Mean coverage mask: ", mean_coverage_list_mask[c])
    print("Mean coverage square: ", mean_coverage_list_square[c])
    print("")







