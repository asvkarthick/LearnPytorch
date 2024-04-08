import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# Gradient
# MSE = 1 / N * (w * x - y)**2
# dJ/dw = 1 / N * 2x * (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print("Prediction before training : {}".format(forward(5)))

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)
    w -= learning_rate * dw

print("Prediction after training : {}".format(forward(5)))
