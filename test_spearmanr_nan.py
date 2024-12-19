from time import time

import numpy as np
from scipy import stats

def confirm_independent():
    mat = np.array([[1, 2, 3, 4],
                    [1, 3, 5, 6],
                    [-1, 2, -1, 4]], dtype=np.float32).T
    corr = stats.spearmanr(mat, nan_policy='omit').correlation
    print(corr)

    mat[1, 1] = np.nan
    corr = stats.spearmanr(mat, nan_policy='omit').correlation
    print(corr)
    quit()

if __name__ == '__main__':


    size = 10_000_000
    num_nans = 9_000_000
    X = np.random.normal(0, 1, size=size)
    Y = np.random.normal(0, 1, size=size)
    rand_idxs = np.random.choice(range(size), size=num_nans, replace=False)
    X[rand_idxs] = np.nan
    rand_idxs = np.random.choice(range(size), size=num_nans, replace=False)
    Y[rand_idxs] = np.nan

    t_st = time()
    r, _ = stats.spearmanr(X, Y, nan_policy='omit')
    t_needed = time() - t_st
    print(f'scipy omit: {t_needed=:.3f}')

    t_st = time()
    nan_idxs = np.isnan(Y) | np.isnan(X)
    X_ = X[~nan_idxs]
    Y_ = Y[~nan_idxs]
    r_manual, _ = stats.spearmanr(X_, Y_)
    t_needed = time() - t_st
    print(f'manual omit: {t_needed=:.3f}')
    small = 1e-12
    assert np.abs(r - r_manual) < small, f'{r=:.6f}, {r_manual=:.6f}'

    # size = 100_000_000
    # num_nans = 10_000_000
    #   scipy omit: t_needed=60.045
    #   manual omit: t_needed=33.123
    # gap widens if num_nans is a bigger % of size
    #   even if the number of nans is just 1, then it will be about twice
    # if the number of nans is 0 then they're basically the same speed
