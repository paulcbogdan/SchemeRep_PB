import numpy as np

from EEG_fMRI.EEG_fMRI_test import get_hrf
from vendor_rs_fluc_timeseries import wiener_deconvolution

if __name__ == '__main__':
    n = 10000
    x = np.random.normal(0, 1, n)
    x[:-1] += x[1:] * .12
    # y = np.random.normal(0, 1, n)

    y = x * -0.12 + np.random.normal(0, 1, n) * .88
    y[:-1] += y[1:] * .12

    r = np.corrcoef(x, y)[0, 1]
    print(f'{r=:.3f}')

    HRF = get_hrf()

    x_ = wiener_deconvolution(x, HRF)
    y_ = wiener_deconvolution(y, HRF)
    r_ = np.corrcoef(x_, y_)[0, 1]
    print(f'{r_=:.3f}')
    print('-')

    autocorr_x = np.corrcoef(x[:-1], x[1:])[0, 1]
    print(f'{autocorr_x=:.3f}')
    autocorr_y = np.corrcoef(y[:-1], y[1:])[0, 1]
    print(f'{autocorr_y=:.3f}')

    dif = np.abs(x - y)
    dif_ = np.abs(x_ - y_)
    autocorr_dif = np.corrcoef(dif[:-1], dif[1:])[0, 1]
    print(f'{autocorr_dif=:.3f}')
    autocorr_dif_ = np.corrcoef(dif_[:-1], dif_[1:])[0, 1]
    print(f'{autocorr_dif_=:.3f}')
