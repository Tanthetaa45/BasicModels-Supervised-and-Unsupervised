import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
linear_model.LinearRegression()
reg.coef_
np.array([0.5, 0.5])