import numpy as np
import scipy.stats as stats




if __name__ == '__main__':

    n = 100000
    t_end = 1
    sigma_mag = 10
    amplitude_mag = 1
    sigma0 = np.random.normal(0, sigma_mag, n)
    # t = np.linspace(0, t_end, n)
    t = np.random.uniform(0, t_end, n)
    amplitude = abs(np.random.normal(0, amplitude_mag, n))
    sigma1 = np.random.normal(0, 1, n)

    x = np.cos(2 * np.pi * t) * amplitude + sigma0
    # np.random.shuffle(x)
    y = np.cos(2 * np.pi * t + np.pi ) * amplitude + sigma1
    # y = -1 * x + sigma1
    # np.random.shuffle(y)

    x = stats.zscore(x)
    y = stats.zscore(y)
    abs_dif = np.abs(x - y)

    corr = stats.pearsonr # pearson correlation yields higher vals than spearman
    r_abs, _ = corr(abs_dif, amplitude)
    print(f'{r_abs=:.4f}')

    prod = x * y
    r_prod, _ = corr(prod, amplitude)
    print(f'{r_prod=:.4f}')

    abs_x_prod, _ = corr(abs_dif, prod)
    print(f'{abs_x_prod=:.4f}')