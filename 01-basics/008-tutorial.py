import torch
import numpy as np

x = np.ones(5)
y = torch.from_numpy(x)
print("X = ", x)
print("Y = ", y)
print("type(x) = ", type(x))
print("type(y) = ", type(y))
