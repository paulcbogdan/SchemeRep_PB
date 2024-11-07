import numpy as np

from Study3.run_analysis_Study3 import get_hrf
from vendor_rs_fluc_timeseries import wiener_deconvolution

def roberto_test():
    x = np.random.normal(0, 1, 200*60)
    y = np.random.normal(0, 1, 200*60)
    max_r = 0
    for i in range(x.shape[0]-10):
        r = np.corrcoef(x[i:i+10], y[i:i+10])[0, 1]
        if r > max_r:
            max_r = r
            print(f'{max_r=:.3f}')
    quit()


if __name__ == '__main__':
    roberto_test()
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
