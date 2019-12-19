import numpy as np
import math


class Feature:

    def __init__(self, dtype, name=None):

        self.feature = np.array(1, dtype)

        self.feature_type = dtype

        if(name is not None):
            self.name = name
