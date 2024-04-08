import torch

x = torch.rand(4, 4)
print("X : \n{}".format(x))
y = x.view(16)
print("Y : \n{}".format(y))
z = x.view(-1, 2)
print("Z : \n{}".format(z))
print("Size of x = ", x.size())
print("Size of y = ", y.size())
print("Size of z = ", z.size())
