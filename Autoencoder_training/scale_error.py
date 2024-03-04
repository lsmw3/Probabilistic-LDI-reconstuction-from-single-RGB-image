import numpy as np
import torch
import os

idx_list = [6, 13, 21]

top_percentile = 90
bottom_percentile = 2

mean = 2.450134528114036
std = 1.1005151495167442

loss = torch.nn.MSELoss()

folder_list_with_ldi_aug = os.listdir("./logs/100TimesTest_with_ldi_aug")
files_list_with_ldi_aug = []

for folder in folder_list_with_ldi_aug:
    files_list_with_ldi_aug.append(f"./logs/100TimesTest_with_ldi_aug/{folder}/images/test")

errors_with_aug = []
scale_with_aug = []
for idx in idx_list:
    error_per_idx =[]
    scale_per_idx = []
    for files in files_list_with_ldi_aug:
        error_per_batch = []
        scale_per_batch = []
        
        gt = np.load('{}/inputs_b-{:06}.npy'.format(files, idx))
        sample = np.load('{}/samples_b-{:06}.npy'.format(files, idx))

        gt = gt * std + mean
        sample = sample * std + mean

        for i in range(gt.shape[0]):
            size_gt = np.max(gt[i]) - np.min(gt[i][np.where(gt[i] != 0)])
            size_sample = np.percentile(sample[i].reshape(-1), top_percentile) - np.percentile(sample[i].reshape(-1), bottom_percentile)
            # sample_idx = np.argsort(sample[i].reshape(-1))
            # size_sample = sample[i].reshape(-1)[sample_idx[-200]] - sample[i].reshape(-1)[sample_idx[199]]

            scale = size_sample / size_gt
            scale_per_batch.append(scale) # shape scale_per_batch [4,]

        gt = torch.tensor(gt)
        sample = torch.tensor(sample)

        for i in range(gt.shape[0]):
            error_per_batch.append(loss(gt[i], sample[i] / scale_per_batch[i])) # shape error_per_batch [4,], distortion error

        error_per_idx.append(error_per_batch) # shape error_per_idx [100, 4]
        scale_per_idx.append(scale_per_batch) # shape scale_per_idx [100, 4]
    
    errors_with_aug.append(error_per_idx)
    scale_with_aug.append(scale_per_idx)

errors_with_aug = np.array(errors_with_aug) # shape error_with_aug [3, 100, 4]
scale_with_aug = np.array(scale_with_aug) # shape scale_with_aug [3, 100, 4]

mean_error_with_aug = []
var_error_with_aug = []
for i in range(errors_with_aug.shape[0]):
    mean_error_with_aug.append(np.mean(errors_with_aug[i], axis=0))
    var_error_with_aug.append(np.var(errors_with_aug[i], axis=0))

mean_error_with_aug = np.array(mean_error_with_aug) # shape mean_scale_with_aug [3, 4]
var_error_with_aug = np.array(var_error_with_aug) # shape var_scale_with_aug [3, 4]

print("Mean error with ldi aug")
print(mean_error_with_aug)

print("Var error with ldi aug")
print(var_error_with_aug)

mean_of_errormeans_with_ldi_aug = np.mean(mean_error_with_aug)
var_of_errormeans_with_ldi_aug = np.var(mean_error_with_aug)

print("Mean of error means with ldi aug", mean_of_errormeans_with_ldi_aug)
print("Var of error means with ldi aug", var_of_errormeans_with_ldi_aug)

print("-------------------------")

mean_scale_with_aug = []
var_scale_with_aug = []
for i in range(scale_with_aug.shape[0]):
    mean_scale_with_aug.append(np.mean(scale_with_aug[i], axis=0))
    var_scale_with_aug.append(np.var(scale_with_aug[i], axis=0))

mean_scale_with_aug = np.array(mean_scale_with_aug) # shape mean_scale_with_aug [3, 4]
var_scale_with_aug = np.array(var_scale_with_aug) # shape var_scale_with_aug [3, 4]

print("Mean scale with ldi aug")
print(mean_scale_with_aug)

print("Var scale with ldi aug")
print(var_scale_with_aug)

mean_of_scalevar_with_ldi_aug = np.mean(var_scale_with_aug)
var_of_scalevar_with_ldi_aug = np.var(var_scale_with_aug)

print("Mean of scale var with ldi aug", mean_of_scalevar_with_ldi_aug)
print("Var of scale var with ldi aug", var_of_scalevar_with_ldi_aug)


print("---------------------------------------------------------------------")


folder_list_no_ldi_aug = os.listdir("./logs/100TimesTest_no_ldi_aug")
files_list_no_ldi_aug = []

for folder in folder_list_no_ldi_aug:
    files_list_no_ldi_aug.append(f"./logs/100TimesTest_no_ldi_aug/{folder}/images/test")

errors_no_aug = []
scale_no_aug = []
for idx in idx_list:
    error_per_idx =[]
    scale_per_idx = []
    for files in files_list_no_ldi_aug:
        error_per_batch = []
        scale_per_batch = []
        
        gt = np.load('{}/inputs_b-{:06}.npy'.format(files, idx))
        sample = np.load('{}/samples_b-{:06}.npy'.format(files, idx))

        gt = gt * std + mean
        sample = sample * std + mean

        for i in range(gt.shape[0]):
            size_gt = np.max(gt[i]) - np.min(gt[i][np.where(gt[i] != 0)])
            size_sample = np.percentile(sample[i].reshape(-1), top_percentile) - np.percentile(sample[i].reshape(-1), bottom_percentile)
            # sample_idx = np.argsort(sample[i].reshape(-1))
            # size_sample = sample[i].reshape(-1)[sample_idx[-200]] - sample[i].reshape(-1)[sample_idx[199]]

            scale = size_sample / size_gt
            scale_per_batch.append(scale) # shape scale_per_batch [4,]

        gt = torch.tensor(gt)
        sample = torch.tensor(sample)

        for i in range(gt.shape[0]):
            error_per_batch.append(loss(gt[i], sample[i] / scale_per_batch[i])) # shape error_per_batch [4,], distortion error

        error_per_idx.append(error_per_batch) # shape error_per_idx [100, 4]
        scale_per_idx.append(scale_per_batch) # shape scale_per_idx [100, 4]
    
    errors_no_aug.append(error_per_idx)
    scale_no_aug.append(scale_per_idx)

errors_no_aug = np.array(errors_no_aug) # shape error_no_aug [3, 100, 4]
scale_no_aug = np.array(scale_no_aug) # shape scale_no_aug [3, 100, 4]

mean_error_no_aug = []
var_error_no_aug = []
for i in range(errors_no_aug.shape[0]):
    mean_error_no_aug.append(np.mean(errors_no_aug[i], axis=0))
    var_error_no_aug.append(np.var(errors_no_aug[i], axis=0))

mean_error_no_aug = np.array(mean_error_no_aug) # shape mean_scale_no_aug [3, 4]
var_error_no_aug = np.array(var_error_no_aug) # shape var_scale_no_aug [3, 4]

print("Mean error no ldi aug")
print(mean_error_no_aug)

print("Var error no ldi aug")
print(var_error_no_aug)

mean_of_errormeans_no_ldi_aug = np.mean(mean_error_no_aug)
var_of_errormeans_no_ldi_aug = np.var(mean_error_no_aug)

print("Mean of error means no ldi aug", mean_of_errormeans_no_ldi_aug)
print("Var of error means no ldi aug", var_of_errormeans_no_ldi_aug)

print("-------------------------")

mean_scale_no_aug = []
var_scale_no_aug = []
for i in range(scale_no_aug.shape[0]):
    mean_scale_no_aug.append(np.mean(scale_no_aug[i], axis=0))
    var_scale_no_aug.append(np.var(scale_no_aug[i], axis=0))

mean_scale_no_aug = np.array(mean_scale_no_aug) # shape mean_scale_no_aug [3, 4]
var_scale_no_aug = np.array(var_scale_no_aug) # shape var_scale_no_aug [3, 4]

print("Mean scale no ldi aug")
print(mean_scale_no_aug)

print("Var scale no ldi aug")
print(var_scale_no_aug)

mean_of_scalevar_no_ldi_aug = np.mean(var_scale_no_aug)
var_of_scalevar_no_ldi_aug = np.var(var_scale_no_aug)

print("Mean of scale var no ldi aug", mean_of_scalevar_no_ldi_aug)
print("Var of scale var no ldi aug", var_of_scalevar_no_ldi_aug)


print("---------------------------------------------------------------------")


folder_list_ldi_aug_larger_interval = os.listdir("./logs/100TimesTest_ldi_aug_larger_interval")
files_list_ldi_aug_larger_interval = []

for folder in folder_list_ldi_aug_larger_interval:
    files_list_ldi_aug_larger_interval.append(f"./logs/100TimesTest_ldi_aug_larger_interval/{folder}/images/test")

errors_ldi_aug_larger_interval = []
scale_ldi_aug_larger_interval = []
for idx in idx_list:
    error_per_idx =[]
    scale_per_idx = []
    for files in files_list_ldi_aug_larger_interval:
        error_per_batch = []
        scale_per_batch = []
        
        gt = np.load('{}/inputs_b-{:06}.npy'.format(files, idx))
        sample = np.load('{}/samples_b-{:06}.npy'.format(files, idx))

        gt = gt * std + mean
        sample = sample * std + mean

        for i in range(gt.shape[0]):
            size_gt = np.max(gt[i]) - np.min(gt[i][np.where(gt[i] != 0)])
            size_sample = np.percentile(sample[i].reshape(-1), top_percentile) - np.percentile(sample[i].reshape(-1), bottom_percentile)
            # sample_idx = np.argsort(sample[i].reshape(-1))
            # size_sample = sample[i].reshape(-1)[sample_idx[-200]] - sample[i].reshape(-1)[sample_idx[199]]

            scale = size_sample / size_gt
            scale_per_batch.append(scale) # shape scale_per_batch [4,]

        gt = torch.tensor(gt)
        sample = torch.tensor(sample)

        for i in range(gt.shape[0]):
            error_per_batch.append(loss(gt[i], sample[i] / scale_per_batch[i])) # shape error_per_batch [4,], distortion error

        error_per_idx.append(error_per_batch) # shape error_per_idx [100, 4]
        scale_per_idx.append(scale_per_batch) # shape scale_per_idx [100, 4]
    
    errors_ldi_aug_larger_interval.append(error_per_idx)
    scale_ldi_aug_larger_interval.append(scale_per_idx)

errors_ldi_aug_larger_interval = np.array(errors_ldi_aug_larger_interval) # shape error_with_aug [3, 100, 4]
scale_ldi_aug_larger_interval = np.array(scale_ldi_aug_larger_interval) # shape scale_with_aug [3, 100, 4]

mean_error_ldi_aug_larger_interval = []
var_error_ldi_aug_larger_interval = []
for i in range(errors_ldi_aug_larger_interval.shape[0]):
    mean_error_ldi_aug_larger_interval.append(np.mean(errors_ldi_aug_larger_interval[i], axis=0))
    var_error_ldi_aug_larger_interval.append(np.var(errors_ldi_aug_larger_interval[i], axis=0))

mean_error_ldi_aug_larger_interval = np.array(mean_error_ldi_aug_larger_interval) # shape mean_scale_with_aug [3, 4]
var_error_ldi_aug_larger_interval = np.array(var_error_ldi_aug_larger_interval) # shape var_scale_with_aug [3, 4]

print("Mean error ldi aug larger interval")
print(mean_error_ldi_aug_larger_interval)

print("Var error ldi aug larger interval")
print(var_error_ldi_aug_larger_interval)

mean_of_errormeans_ldi_aug_larger_interval = np.mean(mean_error_ldi_aug_larger_interval)
var_of_errormeans_ldi_aug_larger_interval = np.var(mean_error_ldi_aug_larger_interval)

print("Mean of error means ldi aug larger interval", mean_of_errormeans_ldi_aug_larger_interval)
print("Var of error means ldi aug larger interval", var_of_errormeans_ldi_aug_larger_interval)

print("-------------------------")

mean_scale_ldi_aug_larger_interval = []
var_scale_ldi_aug_larger_interval = []
for i in range(scale_ldi_aug_larger_interval.shape[0]):
    mean_scale_ldi_aug_larger_interval.append(np.mean(scale_ldi_aug_larger_interval[i], axis=0))
    var_scale_ldi_aug_larger_interval.append(np.var(scale_ldi_aug_larger_interval[i], axis=0))

mean_scale_ldi_aug_larger_interval = np.array(mean_scale_ldi_aug_larger_interval) # shape mean_scale_with_aug [3, 4]
var_scale_ldi_aug_larger_interval = np.array(var_scale_ldi_aug_larger_interval) # shape var_scale_with_aug [3, 4]

print("Mean scale ldi aug larger interval")
print(mean_scale_ldi_aug_larger_interval)

print("Var scale ldi aug larger interval")
print(var_scale_ldi_aug_larger_interval)

mean_of_scalevar_ldi_aug_larger_interval = np.mean(var_scale_ldi_aug_larger_interval)
var_of_scalevar_ldi_aug_larger_interval = np.var(var_scale_ldi_aug_larger_interval)

print("Mean of scale var ldi aug larger interval", mean_of_scalevar_ldi_aug_larger_interval)
print("Var of scale var ldi aug larger interval", var_of_scalevar_ldi_aug_larger_interval)


print("---------------------------------------------------------------------")


folder_list_ldi_aug_neighbor_loss = os.listdir("./logs/100TimesTest_neighbor_loss")
files_list_ldi_aug_neighbor_loss = []

for folder in folder_list_ldi_aug_neighbor_loss:
    files_list_ldi_aug_neighbor_loss.append(f"./logs/100TimesTest_neighbor_loss/{folder}/images/test")

errors_ldi_aug_neighbor_loss = []
scale_ldi_aug_neighbor_loss = []
for idx in idx_list:
    error_per_idx =[]
    scale_per_idx = []
    for files in files_list_ldi_aug_neighbor_loss:
        error_per_batch = []
        scale_per_batch = []
        
        gt = np.load('{}/inputs_b-{:06}.npy'.format(files, idx))
        sample = np.load('{}/samples_b-{:06}.npy'.format(files, idx))

        gt = gt * std + mean
        sample = sample * std + mean

        for i in range(gt.shape[0]):
            size_gt = np.max(gt[i]) - np.min(gt[i])
            size_sample = np.percentile(sample[i].reshape(-1), top_percentile) - np.percentile(sample[i].reshape(-1), bottom_percentile)
            # sample_idx = np.argsort(sample[i].reshape(-1))
            # size_sample = sample[i].reshape(-1)[sample_idx[-200]] - sample[i].reshape(-1)[sample_idx[199]]

            scale = size_sample / size_gt
            scale_per_batch.append(scale) # shape scale_per_batch [4,]

        gt = torch.tensor(gt)
        sample = torch.tensor(sample)

        for i in range(gt.shape[0]):
            error_per_batch.append(loss(gt[i], sample[i] / scale_per_batch[i])) # shape error_per_batch [4,], distortion error

        error_per_idx.append(error_per_batch) # shape error_per_idx [100, 4]
        scale_per_idx.append(scale_per_batch) # shape scale_per_idx [100, 4]
    
    errors_ldi_aug_neighbor_loss.append(error_per_idx)
    scale_ldi_aug_neighbor_loss.append(scale_per_idx)

errors_ldi_aug_neighbor_loss = np.array(errors_ldi_aug_neighbor_loss) # shape error_with_aug [3, 100, 4]
scale_ldi_aug_neighbor_loss = np.array(scale_ldi_aug_neighbor_loss) # shape scale_with_aug [3, 100, 4]

mean_error_ldi_aug_neighbor_loss = []
var_error_ldi_aug_neighbor_loss = []
for i in range(errors_ldi_aug_neighbor_loss.shape[0]):
    mean_error_ldi_aug_neighbor_loss.append(np.mean(errors_ldi_aug_neighbor_loss[i], axis=0))
    var_error_ldi_aug_neighbor_loss.append(np.var(errors_ldi_aug_neighbor_loss[i], axis=0))

mean_error_ldi_aug_neighbor_loss = np.array(mean_error_ldi_aug_neighbor_loss) # shape mean_scale_with_aug [3, 4]
var_error_ldi_aug_neighbor_loss = np.array(var_error_ldi_aug_neighbor_loss) # shape var_scale_with_aug [3, 4]

print("Mean error ldi aug neighbor loss")
print(mean_error_ldi_aug_neighbor_loss)

print("Var error ldi aug neighbor loss")
print(var_error_ldi_aug_neighbor_loss)

mean_of_errormeans_ldi_aug_neighbor_loss = np.mean(mean_error_ldi_aug_neighbor_loss)
var_of_errormeans_ldi_aug_neighbor_loss = np.var(mean_error_ldi_aug_neighbor_loss)

print("Mean of error means ldi aug neighbor loss", mean_of_errormeans_ldi_aug_neighbor_loss)
print("Var of error means ldi aug neighbor loss", var_of_errormeans_ldi_aug_neighbor_loss)

print("-------------------------")

mean_scale_ldi_aug_neighbor_loss = []
var_scale_ldi_aug_neighbor_loss = []
for i in range(scale_ldi_aug_neighbor_loss.shape[0]):
    mean_scale_ldi_aug_neighbor_loss.append(np.mean(scale_ldi_aug_neighbor_loss[i], axis=0))
    var_scale_ldi_aug_neighbor_loss.append(np.var(scale_ldi_aug_neighbor_loss[i], axis=0))

mean_scale_ldi_aug_neighbor_loss = np.array(mean_scale_ldi_aug_neighbor_loss) # shape mean_scale_with_aug [3, 4]
var_scale_ldi_aug_neighbor_loss = np.array(var_scale_ldi_aug_neighbor_loss) # shape var_scale_with_aug [3, 4]

print("Mean scale ldi aug neighbor loss")
print(mean_scale_ldi_aug_neighbor_loss)

print("Var scale ldi aug neighbor loss")
print(var_scale_ldi_aug_neighbor_loss)

mean_of_scalevar_ldi_aug_neighbor_loss = np.mean(var_scale_ldi_aug_neighbor_loss)
var_of_scalevar_ldi_aug_neighbor_loss = np.var(var_scale_ldi_aug_neighbor_loss)

print("Mean of scale var ldi aug neighbor loss", mean_of_scalevar_ldi_aug_neighbor_loss)
print("Var of scale var ldi aug neighbor loss", var_of_scalevar_ldi_aug_neighbor_loss)


"""
print("Difference of mean of errors between with ldi aug and no ldi aug")
print(mean_error_with_aug - mean_error_no_aug)

print("Difference of mean of scale between with ldi aug and no ldi aug")
print(mean_scale_with_aug - mean_scale_no_aug)

print("Difference of var of errors between with ldi aug and no ldi aug")
print(var_error_with_aug - var_error_no_aug)

print("Difference of var of scale between with ldi aug and no ldi aug")
print(var_scale_with_aug - var_scale_no_aug)
"""
