import torch

print(f"{torch.cuda.is_available()=}")

f = torch.nn.Conv2d(3, 8, 3, device="cuda")
X = torch.randn(2, 3, 4, 4, device="cuda")

Y = X @ X
print(f"{Y.shape=}")
print("matrix multiply works")

Y = f(X)
print(f"{Y.shape=}")
print("Conv2d works")