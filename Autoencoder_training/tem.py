import torch

a = torch.load("./models/pre_trained_weights/kl-f4/best_model.pth", map_location="cpu")
print(a)