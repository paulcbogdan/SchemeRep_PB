import numpy as np
import scipy.stats as stats

from Study3.run_analysis_Study3 import get_hrf


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
    sigma_mag = 2
    amplitude_mag = 10
    sigma0 = np.random.normal(0, sigma_mag, n)
    t = np.linspace(0, t_end, n)
    # t = np.random.uniform(0, t_end, n)
    amplitude = abs(np.random.normal(0, amplitude_mag, n))
    # amplitude = 10
    sigma1 = np.random.normal(0, sigma_mag, n)
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

    # plt.plot(x[:100], linewidth=0.5)
    # plt.plot(y[:100], linewidth=0.5)
    # plt.show()

    avg = 100
    x_ = np.mean(x.reshape(-1, avg), axis=1)
    y_ = np.mean(y.reshape(-1, avg), axis=1)

    x = conv_quick(x)
    y = conv_quick(y)

    # plt.plot(x[:100], linewidth=0.5)
    # plt.plot(y[:100], linewidth=0.5)
    # plt.show()

    amplitude_ = np.mean(amplitude.reshape(-1, avg), axis=1)
    amplitude_ = conv_quick(amplitude_)

    r_x_y, _ = stats.pearsonr(x_, y_)
    print(f'x x y: {r_x_y:.4f}')

    x_ = stats.zscore(x_)
    y_ = stats.zscore(y_)
    abs_dif_ = np.abs(x_ - y_)
    print(amplitude_.shape)
    print(abs_dif_.shape)

    corr = stats.spearmanr # pearson correlation yields higher vals than spearman
    r_abs, _ = corr(abs_dif_, amplitude_)
    print(f'{r_abs=:.4f}')

    # prod = x * y
    # r_prod, _ = corr(prod, amplitude)
    # print(f'{r_prod=:.4f}')
    #
    # abs_x_prod, _ = corr(abs_dif, prod)
    # print(f'{abs_x_prod=:.4f}')