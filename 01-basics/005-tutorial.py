import torch

x = torch.rand(3, 3)
print("X : \n{}".format(x))
print("X[:, 1] : {}".format(x[:, 1]))
print("X[1, :] : {}".format(x[1, :]))
print("X[2, :] : {}".format(x[2, :]))
print("X[1, 1] : {}".format(x[1, 1]))
print("X[1, 1] : {}".format(x[1, 1].item()))
