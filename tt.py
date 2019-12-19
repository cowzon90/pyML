import numpy as np

x0 = np.ones([100, 1])
x1 = np.random.rand(100).reshape([100, 1]) * 1000.0 % 300.0 + 100.0

x = np.hstack((x0, x1))

print(x0.shape)
print(x1.shape)
print(x.shape)


print(np.dot(x0.transpose(), x1))