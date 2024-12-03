import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.random.multivariate_normal([0, 0], cov=[[1, 0.5], [0.5, 1]], size=10_000)
# x = (x > 0) * 2 - 1

mixup = ((x[:, 0] > 0) & (x[:, 1] < 0)) | ((x[:, 0] < 0) & (x[:, 1] > 0))
# mixup = ((x[:, 0] < 0) & (x[:, 1] < 0)) | ((x[:, 0] > 0) & (x[:, 1] > 0))

# mixup = x[:, 0] != x[:, 1]
x_mixup = x[mixup]
r, p = stats.pearsonr(x_mixup[:, 0], x_mixup[:, 1])
print(f'{r=:.3f}, {p=:.3f}')

plt.scatter(x_mixup[:, 0], x_mixup[:, 1], alpha=.5)
plt.show()




