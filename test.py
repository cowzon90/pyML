import numpy as np
from LinearRegression import pyLinearRegression as linear

def test():
    a = np.arange(25).reshape([5, 5])

    b = np.zeros([5, 1])

    print(a)

    print(b)

    print("stack")

    c = np.hstack((b, a))
    print(c)


print("lr")
x0 = np.ones([1000, 1])
x1 = np.random.rand(1000).reshape([1000, 1]) * 1000.0 % 10.0 + 10.0 # 10 to 20
y = np.random.rand(1000).reshape([1000, 1]) * 1000.0 % 100.0 + 100.0 # 100 to 200

x = np.hstack((x0, x1))
# print(x1)
# print(y)

arrangex1 = np.arange(100).reshape([100, 1])
arrangey = np.arange(100).reshape([100, 1]) * 2


iter = 2000
learningrate = 0.00005

lr = linear.LinearRegression()
lr.setdata(arrangex1, arrangey)
lr.do_linearRegression(iter, learningrate, scaling=False)
