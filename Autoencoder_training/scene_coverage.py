import numpy as np
import torch
import os

mean = 2.450134528114036
std = 1.1005151495167442

three_layer_gt_coverage = 0.7215836821595127

top_percentile = 90
bottom_percentile = 2

__MODE__ = 'train_on_7k'
seed = 163

zoedepth_out_path = "../ZoeDepth/output/zoedepth_out.npy"
zoedepth_out = np.load(zoedepth_out_path) # shape 20 x 1 x 256 x 256

sample_coverage = []
zoedepth_coverage = []
intersection_part_first_layer = []
intersection_part_second_layer = []
intersection_part_third_layer = []
sample_coverage_first_layer = []
sample_coverage_second_layer = []
sample_coverage_third_layer = []
for idx in range(5):
    inputs_path = "./logs/{}_seed{}/images/test/inputs_b-{:06}.npy".format(__MODE__, seed, idx)
    samples_path = "./logs/{}_seed{}/images/test/samples_b-{:06}.npy".format(__MODE__, seed, idx)

    gt = np.load(inputs_path) # shape 4 x 3 x 256 x 256
    sample = np.load(samples_path) # shape 4 x 3 x 256 x 256

    gt = gt * std + mean
    sample = sample * std + mean

    for i in range(gt.shape[0]):
        gt_valid_region = np.where(gt[i] != 0, 1, 0)
        gt_coverage_i = np.sum(gt_valid_region)

        zeodepth_valid_region = np.where(zoedepth_out[4 * idx + i] != 0, 1, 0)
        zeodepth_coverage_i = np.sum(zeodepth_valid_region)

        sample_coverage_i = 0
        intersection_i = 0
        for j in range(sample.shape[1]):
            if j == 0:
                sample_valid_region = np.where(sample[i][j] != 0, 1, 0)
                sample_intersection = np.where((sample_valid_region * gt_valid_region[j] == 1), 1, 0)
                intersection_part_first_layer.append(np.sum(sample_intersection) / np.sum(gt_valid_region[j]))
                sample_coverage_first_layer.append(np.sum(sample_valid_region) / np.sum(gt_valid_region[j]))
            else:
                sample_valid_region = np.where((sample[i][j] > sample[i][j - 1]) & (sample[i][j] > sample[i][0]), 1, 0)
                sample_intersection = np.where((sample_valid_region * gt_valid_region[j] == 1), 1, 0)
                if j == 1:
                    intersection_part_second_layer.append(np.sum(sample_intersection) / np.sum(gt_valid_region[j]))
                    sample_coverage_second_layer.append(np.sum(sample_valid_region) / np.sum(gt_valid_region[j]))
                elif j == 2:
                    intersection_part_third_layer.append(np.sum(sample_intersection) / np.sum(gt_valid_region[j]))
                    sample_coverage_third_layer.append(np.sum(sample_valid_region) / np.sum(gt_valid_region[j]))
            sample_coverage_i += np.sum(sample_valid_region)

        sample_coverage.append(three_layer_gt_coverage * sample_coverage_i / gt_coverage_i)
        zoedepth_coverage.append(three_layer_gt_coverage * zeodepth_coverage_i / gt_coverage_i)

sample_coverage = np.array(sample_coverage)
zoedepth_coverage = np.array(zoedepth_coverage)
intersection_part_first_layer = np.array(intersection_part_first_layer)
intersection_part_second_layer = np.array(intersection_part_second_layer)
intersection_part_third_layer = np.array(intersection_part_third_layer)
sample_coverage_first_layer = np.array(sample_coverage_first_layer)
sample_coverage_second_layer = np.array(sample_coverage_second_layer)
sample_coverage_third_layer = np.array(sample_coverage_third_layer)

print(f"sample coverage {__MODE__} seed {seed}")
print(sample_coverage)
print("average sample coverage")
print(np.mean(sample_coverage))
print("")

print(f"zoedepth coverage {__MODE__} seed {seed}")
print(zoedepth_coverage)
print("average zoedepth coverage")
print(np.mean(zoedepth_coverage))
print("")

print("intersection part first layer")
print(intersection_part_first_layer)
print("average intersection part first layer")
print(np.mean(intersection_part_first_layer))
print("average first layer coverage")
print(np.mean(sample_coverage_first_layer))
print("")

print("intersection part second layer")
print(intersection_part_second_layer)
print("average intersection part second layer")
print(np.mean(intersection_part_second_layer))
print("average secondt layer coverage")
print(np.mean(sample_coverage_second_layer))
print("")

print("intersection part third layer")
print(intersection_part_third_layer)
print("average intersection part third layer")
print(np.mean(intersection_part_third_layer))
print("average third layer coverage")
print(np.mean(sample_coverage_third_layer))