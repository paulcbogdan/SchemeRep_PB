import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

from EEG_fMRI.EEG_fMRI_test import get_hrf


def conv_quick(ts):
    import scipy.ndimage as ndimage
    HRF = get_hrf()
    # for i in range(EEG_fluc.shape[1]):

    ts_ = ndimage.convolve1d(ts, HRF, mode='nearest',
                                  origin=-HRF.shape[0] // 2, axis=0)
    return ts_

if __name__ == '__main__':

    n = 100000
    t_end = 10000
    sigma_mag = 10
    amplitude_mag = 1
    sigma0 = np.random.normal(0, sigma_mag, n)
    t = np.linspace(0, t_end, n)
    # t = np.random.uniform(0, t_end, n)
    amplitude = abs(np.random.normal(0, amplitude_mag, n))
    # amplitude = 10
    sigma1 = np.random.normal(0, 1, n)
    # sigma0 = 0

    x = np.cos(2 * np.pi * t) * amplitude + sigma0
    # print(x.shape)
    # quit()
    # plt.plot(x[:100], linewidth=0.5)
    # plt.show()
    # quit()

    # np.random.shuffle(x)
    y = np.cos(2 * np.pi * t + np.pi ) * amplitude + sigma1

    # y = -1 * x + sigma1
    # np.random.shuffle(y)

    x = conv_quick(x)
    y = conv_quick(y)
    amplitude = conv_quick(amplitude)

    x = stats.zscore(x)
    y = stats.zscore(y)
    abs_dif = np.abs(x - y)

    corr = stats.pearsonr # pearson correlation yields higher vals than spearman
    r_abs, _ = corr(abs_dif, amplitude)
    print(f'{r_abs=:.4f}')

    # prod = x * y
    # r_prod, _ = corr(prod, amplitude)
    # print(f'{r_prod=:.4f}')
    #
    # abs_x_prod, _ = corr(abs_dif, prod)
    # print(f'{abs_x_prod=:.4f}')