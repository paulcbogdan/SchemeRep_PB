

import numpy as np
from sklearn import datasets
from scipy.stats import f


def TwoSampleT2Test(X, Y):
    nx, p = X.shape
    # print(f'{X.shape=}')
    ny, _ = Y.shape
    # print(f'{Y.shape=}')
    # print(f'{Y=}')
    # quit()
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    Sx = np.cov(X, rowvar=False)
    Sy = np.cov(Y, rowvar=False)
    S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
    t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
    statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
    F = f(p, nx+ny-p-1)
    p_value = 1 - F.cdf(statistic)
    print(f"Test statistic: {statistic}\nDegrees of freedom: {p} and {nx+ny-p-1}\np-value: {p_value}")
    return statistic, p_value


if __name__ == '__main__':
    iris = datasets.load_iris()
    versicolor = iris.data[iris.target==1, :2]
    virginica = iris.data[iris.target==2, :2]
    print(iris)
    TwoSampleT2Test(versicolor, virginica)












