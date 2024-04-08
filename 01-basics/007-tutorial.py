import torch
import numpy as np

x = torch.ones(5)
y = x.numpy()
print("X = ", x)
print("Y = ", y)
print("type(x) = ", type(x))
print("type(y) = ", type(y))
