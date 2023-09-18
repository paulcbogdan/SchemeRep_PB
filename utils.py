from collections import defaultdict

import numpy as np
from scipy import stats as stats



def regress_out(x, y):
    x = np.array(x)
    y = np.array(y)
    nans = np.isnan(x) | np.isnan(y)
    if len(x[~nans]) < 3:
        return y
    b, m, r, p, er = stats.linregress(x[~nans], y[~nans])
    return y - x*b

def regress_out_multi(X, y):
    for x in X:
        y = regress_out(x, y)
    return y

def stdize(v, axis=None):
    return (v - np.mean(v, axis=axis)) / np.std(v, axis=axis)


def nan_ar(shape):
    return np.full(shape, np.nan)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        for key, d_sub in d.items():
            d[key] = defaultdict_to_dict(d_sub)
    return d


def pb_outer(a, b, flat=False):
    # Multiply matrix of vector a (size n) and vector b (size m) to get n x m
    a = np.array(a).T
    b = np.array(b).T
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=1)
    c = a * b
    if flat:
        c = c.reshape(-1, c.shape[1]*c.shape[2])
    return c


def pb_outer_multi(a, b, flat=False):
    # b is a list. For each b:
    #   multiply matrix of vector a (size n) and each vector b (size m) to get n x m
    a = np.array(a)
    b = fill_w_nans(b)
    b = np.array(b)
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=2)
    c = a * b
    if flat:
        c = c.reshape(b.shape[0], -1, c.shape[-2]*c.shape[-1])
    return c


def fill_w_nans(b):
    lens = [len(ROI[0]) for ROI in b]
    max_size = max(lens)
    b_filled = np.full((len(b), len(b[0]), max_size), np.nan)
    for i in range(len(b)):
        for j in range(len(b[i])):
            b_filled[i, j, :len(b[i][j])] = b[i][j]
    return b_filled


def pb_outer_double_multi(a, b, flat=False):
    # a and b are list. For each combination of a and b
    #   multiply matrix of vector a (size n) and each vector b (size m) to get n x m

    a = fill_w_nans(a)
    b = fill_w_nans(b)
    a = np.array(a)
    b = np.array(b)
    a = np.expand_dims(a, axis=1)
    a = np.expand_dims(a, axis=-1) # (ROIs, 1, n_trials, longest_vec_length, 1)
    b = np.expand_dims(b, axis=0)
    b = np.expand_dims(b, axis=3) # (1, ROIs, n_trials, 1, longest_vec_length)
    assert a.shape[2] == b.shape[2], f'n_trials must be the same: {b.shape=}, {a.shape=}'
    c = a * b
    if flat:
        c = c.reshape(a.shape[0], b.shape[1], a.shape[2], c.shape[-2]*c.shape[-1])
    return c


