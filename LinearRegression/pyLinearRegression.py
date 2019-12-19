import numpy as np

class LinearRegression:
    """
    Linear Regression Model
    """
    def __init__(self, data=None, y=None):

        if data is not None:
            self.data = data
            self.initialize_weight()  # init weight
        if y is not None:
            self.y = y

    def hypothesis(self):
        pass

    def setdata(self, data, y):
        """
        set data matrix and y vector
        :param data: data matirx (numpy array)
        :param y: y vector (numpy array)
        :return:
        """
        m, n = data.shape
        self.features = n
        self.datasize = m

        if y.shape[0] != self.datasize:
            raise Exception("data and y is not matched.", y.shape[0], self.datasize)

        bias = np.ones([self.datasize, 1])
        self.data = np.hstack((bias, data))
        self.y = y

        # init weight
        self.initialize_weight()

        print("data summary")
        print("x : ", self.data.shape, self.data.min(), self.data.max())
        print("y : ", self.y.shape, self.y.min(), self.y.max())
        print("weight : ", self.weight.shape)
        print(self.weight)

    def initialize_weight(self, init="rand"):
        """
        initialize weigth (theta)
        :param init: init numpy array
        :return:
        """
        if init is "rand":
            # random
            self.weight = np.random.rand(self.features + 1, 1)
            return self.weight
        elif init is "zero":
            # init zero
            self.weight = np.zeros([self.features + 1, 1])
            return self.weight
        else:
            pass

            # if self.weight.shape is init.shape:
            #     self.weight = init
            #     return self.weight

    def function(self, x_vector):
        """
        function is transpose of theta multiply with x-vector
        :param x_vector: row vector
        :return: function result
        """

        shape_of_weight = self.weight.shape[0]

        if shape_of_weight is not (x_vector.shape[0]):
            raise Exception("shape of weight : ", shape_of_weight, ", x_vector : ", x_vector)

        val = np.dot(self.weight.transpose(), x_vector)
        return val[0]

    def __costfunction(self):
        """
        cost function of linear regression
        :param data: all data (numpy array)
        :param y_vector: y vector
        :return: 1/2m (i = 0 to m-1)Sigma(hypothesis(x) - y)^2)
        """
        val = 0.0
        for height in range(self.data.shape[0]):

            f = self.function(self.data[height, :])
            y = self.y[height, 0]
            val = val + (f-y)*(f-y)

        val = val / (2 * self.datasize)

        return val

    def __gradientDescent(self, data, y, index:int, learningrate):
        """
        Gradient Descent with weight index
        :param index: index of weight
        :return:
        """
        sigma = 0.0
        for height in range(self.datasize):

            # x_vector = self.data[height, :].transpose()
            # shape = x_vector.reshape([2,1])
            # weight = self.weight[index, 0]
            # function_value = self.function(self.data[height, :].transpose())
            # y_value = self.y[height, 0]
            # x_vector_index = self.data[height, index]

            sigma += ((self.function(data[height, :].transpose()) - y[height, 0]) * data[height, index])
            # print(self.data[height, :].transpose())

        val = self.weight[index, 0] - (learningrate / self.datasize * sigma)
        return val

    def do_linearRegression(self, iter:int, learningrate:float, scaling=False):
        """
        do linear regression with iteration number and learning rate.

        :param iter: number of iterator
        :param learningrate: learning rate
        :return:
        """

        print("Linear Regression start.")
        print("Info")
        print("weight")
        print(self.weight)
        print("iteration : ", iter)

        if scaling:
            print("use scaled features")
            self.feature_scaling()
            data = self.scaled_data
            y = self.scaled_y
            # y = self.y
        else:
            data = self.data
            y = self.y

        for i in range(iter):

            weight = np.zeros(self.weight.shape)
            for index in range(0, self.features + 1):
                weight[index, 0] = self.__gradientDescent(data, y, index, learningrate)
                print("weight {0} change {1} to {2}".format(index, self.weight[index,0], weight[index,0]))

            self.weight = weight
            print("cost function {0} is {1}".format(i, self.__costfunction()))

    def normal_equation(self):
        """
        Normal Equation of linear regression
        theta = (x^t dot x)^-1 dot x^t dot y
        :return: theta
        """
        inverse = np.linalg.pinv(np.dot(self.data.transpose(), self.data))

        theta = np.dot(np.dot(inverse, self.data.transpose()), self.y)

        return theta

    def feature_scaling(self):
        """
        feature scaling // TODO
        :return:
        """

        self.scaled_data = np.ones([self.data.shape[0], self.data.shape[1]], dtype=np.float64)

        for x in range(1, self.features + 1):

            mean = np.mean(self.data[:, x])
            max = np.amax(self.data[:, x])
            min = np.amin(self.data[:, x])
            print(mean, max, min)
            self.scaled_data[:, x] = ((self.data[:, x] - mean) / (max - min))

        mean = np.mean(self.y)
        max = np.amax(self.y)
        min = np.amin(self.y)

        self.scaled_y = ((self.y - mean) / (max - min))

        return self.scaled_data, self.scaled_y
