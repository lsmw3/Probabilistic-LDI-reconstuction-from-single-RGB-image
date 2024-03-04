import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim

from util import instantiate_from_config

mean = 2.450134528114036
std = 1.1005151495167442

top_percentile = 90
bottom_percentile = 2

__MODE__ = 'train_on_7k'
seed = 163

sigma_1 = 1.25
sigma_2 = 1.25 ** 2
sigma_3 = 1.25 ** 3

mse_loss = torch.nn.MSELoss()

def metrics(a, b, layer_indx):
   if layer_indx == 0:
        # absolute relative error
        rel = np.mean(np.abs(b - a) / a)

        # root mean square error
        rmse = np.sqrt(mse_loss(torch.tensor(b), torch.tensor(a)).numpy())

        # average log10 error
        log10 = np.mean(np.abs(np.log10(b) - np.log10(a)))
                
        # threshold accuracy
        # n == 1
        sigma1_array = np.maximum(b.reshape(-1) / a.reshape(-1), a.reshape(-1) / b.reshape(-1)) < sigma_1 # shape [256x256,]
        sigma1 = (sigma1_array == True).sum() / sigma1_array.size
        # n == 2
        sigma2_array = np.maximum(b.reshape(-1) / a.reshape(-1), a.reshape(-1) / b.reshape(-1)) < sigma_2 # shape [256x256,]
        sigma2 = (sigma2_array == True).sum() / sigma2_array.size
        # n == 3
        sigma3_array = np.maximum(b.reshape(-1) / a.reshape(-1), a.reshape(-1) / b.reshape(-1)) < sigma_3 # shape [256x256,]
        sigma3 = (sigma3_array == True).sum() / sigma3_array.size

        return np.array([rel, rmse, log10, sigma1, sigma2, sigma3])
   else:
       # root mean square error
        rmse = np.sqrt(mse_loss(torch.tensor(b), torch.tensor(a)).numpy())
        return rmse

zoedepth_out_path = "../ZoeDepth/output/zoedepth_out_whole_testset.npy"
zoedepth_out = np.load(zoedepth_out_path) # shape 96 x 1 x 256 x 256

error_zoedepth = []
error_sample_first_layer = []
error_sample_second_layer_union = []
error_sample_second_layer_intersection = []
error_sample_third_layer_union = []
error_sample_third_layer_intersection = []
ssim_zeodepth = []
ssim_first_layer = []
ssim_second_layer_union = []
ssim_second_layer_intersection = []
ssim_third_layer_union = []
ssim_third_layer_intersection= []
for idx in range(24):
    inputs_path = "./logs/{}_seed{}/images/test/inputs_b-{:06}.npy".format(__MODE__, seed, idx)
    samples_path = "./logs/{}_seed{}/images/test/samples_b-{:06}.npy".format(__MODE__, seed, idx)

    gt = np.load(inputs_path) # shape 4 x 3 x 256 x 256
    sample = np.load(samples_path) # shape 4 x 3 x 256 x 256

    gt = gt * std + mean
    sample = sample * std + mean

    
    for i in range(gt.shape[0]):
        # per-layer normalization
        for j in range(gt.shape[1]):
            if j == 0:
                gt_min = np.min(gt[i][j])
                gt_max = np.max(gt[i][j])

                sample_min = np.percentile(sample[i][j], bottom_percentile)
                sample_max = np.percentile(sample[i][j], top_percentile)

                zoedepth_min = np.percentile(zoedepth_out[4 * idx + i], bottom_percentile)
                zoedepth_max = np.percentile(zoedepth_out[4 * idx + i], top_percentile)

                sample_j = sample[i][j] * (gt_max - gt_min) / (sample_max - sample_min) # shape 256 x 256
                zoedepth_j = zoedepth_out[4 * idx + i][0] * (gt_max - gt_min) / (zoedepth_max - zoedepth_min) # shape 256 x 256

                error_sample_first_layer.append(metrics(gt[i][j], sample_j, 0))
                error_zoedepth.append(metrics(gt[i][j], zoedepth_j, 0))

                ssim_first_layer.append(ssim(gt[i][j], sample_j, data_range=max(gt[i][j].max(), sample_j.max()) - min(gt[i][j].min(), sample_j.min())))
                ssim_zeodepth.append(ssim(gt[i][j], zoedepth_j, data_range=max(gt[i][j].max(), zoedepth_j.max()) - min(gt[i][j].min(), zoedepth_j.min())))
            else:
                gt_valid_region = gt[i][j][np.where(gt[i][j] != 0)]
                gt_min = np.min(gt_valid_region)
                gt_max = np.max(gt_valid_region)

                sample_valid_region = sample[i][j][np.where((sample[i][j] > sample[i][j - 1]) & (sample[i][j] > sample[i][0]))]
                sample_min = np.percentile(sample_valid_region, 2)
                sample_max = np.percentile(sample_valid_region, 98)

                sample_j = sample[i][j] * (gt_max - gt_min) / (sample_max - sample_min) # shape 256 x 256
                sample_j[np.where((sample[i][j] < sample[i][j - 1]) | (sample[i][j] < sample[i][0]))] = 0

                gt_union = gt[i][j][np.where(((sample[i][j] > sample[i][j - 1]) & (sample[i][j] > sample[i][0])) | (gt[i][j] != 0))]
                sample_union = sample_j[np.where(((sample[i][j] > sample[i][j - 1]) & (sample[i][j] > sample[i][0])) | (gt[i][j] != 0))]

                gt_intersection = gt[i][j][np.where(sample_j * gt[i][j] > 0)]
                sample_intersection = sample_j[np.where(sample_j * gt[i][j] > 0)]

                if j == 1:
                    error_sample_second_layer_union.append(metrics(gt_union, sample_union, 1))
                    error_sample_second_layer_intersection.append(metrics(gt_intersection, sample_intersection, 1))
                    ssim_second_layer_union.append(ssim(gt_union, sample_union, data_range=max(sample_union.max(), gt_union.max()) - min(sample_union.min(), gt_union.min())))
                    ssim_second_layer_intersection.append(ssim(gt_intersection, sample_intersection, data_range=max(sample_intersection.max(), gt_intersection.max()) - min(sample_intersection.min(), gt_intersection.min())))
                elif j == 2:
                    error_sample_third_layer_union.append(metrics(gt_union, sample_union, 2))
                    error_sample_third_layer_intersection.append(metrics(gt_intersection, sample_intersection, 2))
                    ssim_third_layer_union.append(ssim(gt_union, sample_union, data_range=max(sample_union.max(), gt_union.max()) - min(sample_union.min(), gt_union.min())))
                    ssim_third_layer_intersection.append(ssim(gt_intersection, sample_intersection, data_range=max(sample_intersection.max(), gt_intersection.max()) - min(sample_intersection.min(), gt_intersection.min())))

error_zoedepth = np.array(error_zoedepth)
error_sample_first_layer = np.array(error_sample_first_layer)
error_sample_second_layer_union = np.array(error_sample_second_layer_union)
error_sample_second_layer_intersection = np.array(error_sample_second_layer_intersection)
error_sample_third_layer_union = np.array(error_sample_third_layer_union)
error_sample_third_layer_intersection = np.array(error_sample_third_layer_intersection)

ssim_zeodepth = np.array(ssim_zeodepth)
ssim_first_layer = np.array(ssim_first_layer)
ssim_second_layer_union = np.array(ssim_second_layer_union)
ssim_second_layer_intersection = np.array(ssim_second_layer_intersection)
ssim_third_layer_union = np.array(ssim_third_layer_union)
ssim_third_layer_intersection = np.array(ssim_third_layer_intersection)

print("zoedepth error")
print(np.mean(error_zoedepth, axis=0))

print("")

print("first layer error")
print(np.mean(error_sample_first_layer, axis=0))

print("")

print("second layer error union")
print(np.mean(error_sample_second_layer_union))
print("second layer error intersection")
print(np.mean(error_sample_second_layer_intersection))

print("")

print("third layer error union")
print(np.mean(error_sample_third_layer_union))
print("third layer error intersection")
print(np.mean(error_sample_third_layer_intersection))

print("")

print("zeodepth ssim")
print(ssim_zeodepth.mean())

print("")

print("first layer ssim")
print(ssim_first_layer.mean())

print("")

print("second layer ssim union")
print(ssim_second_layer_union.mean())
print("second layer ssim intersection")
print(ssim_second_layer_intersection.mean())

print("")

print("third layer ssim union")
print(ssim_third_layer_union.mean())
print("third layer ssim intersection")
print(ssim_third_layer_intersection.mean())