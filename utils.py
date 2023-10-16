from collections import defaultdict

import numpy as np
from scipy import stats as stats



def regress_out(x, y):
    x = np.array(x)
    y = np.array(y)
    nans = np.isnan(x) | np.isnan(y)
    n_goods = np.sum(~nans)
    if n_goods < 3:
        # print(f'Warning: not enough non-nan values to regress out: {n_goods=}')
        return y
    b, m, r, p, er = stats.linregress(x[~nans], y[~nans])
    return y - x*b

def regress_out_multi(X, y):
    for x in X:
        y = regress_out(x, y)
    return y

def stdize(v, axis=None, nans=False):
    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    if axis == 2:
        return (v - m(v, axis=axis)[:, :, None]) / s(v, axis=axis)[:, :, None]
    elif axis == 1:
        return (v - m(v, axis=axis)[:, None]) / s(v, axis=axis)[:, None]
    else:
        return (v - m(v, axis=axis)) / s(v, axis=axis)


def nan_ar(shape):
    return np.full(shape, np.nan)


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        for key, d_sub in d.items():
            d[key] = defaultdict_to_dict(d_sub)
    if isinstance(d, list):
        d = np.array(d)
    return d


def pb_outer(a, b, flat=False, tril=False, nan_diag=False):
    # Multiply matrix of vector a (size n) and vector b (size m) to get n x m
    a = np.array(a)
    b = np.array(b)
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=1)
    c = a * b
    if nan_diag:
        assert c.shape[-1] == c.shape[-2], f'c must be square: {c.shape=}'
        c[:, np.arange(c.shape[-1]), np.arange(c.shape[-1])] = np.nan
    if tril:
        c = ndim_tril_flatten(c)
    elif flat:
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

def get_RSA_name(key):
    if key == 'obj':
        key_str = 'Object RSA'
    elif key == 'scn':
        key_str = 'Scene RSA'
    elif key == 'scn_abs':
        key_str = 'Abs-scene RSA'
    elif key == 'dif':
        key_str = 'Difference RSA'
    elif key == 'dif_':
        key_str = 'Difference RSA (regressed)'
    elif key == 'dif_abs':
        key_str = 'Abs-dif RSA'
    elif key == 'dif_abs_':
        key_str = 'Abs-dif RSA (regressed)'
    elif key == 'add':
        key_str = 'Addition RSA'
    elif key == 'add_abs':
        key_str = 'Abs-addition RSA'
    elif key == 'prd':
        key_str = 'Product RSA'
    elif key == 'prd_abs':
        key_str = 'Abs-product RSA'
    else:
        key_str = ''
    return key_str

def make_title_str(pre_str, key, age, early, semantic, cin=None,
                   short=False):
    key_str = get_RSA_name(key)
    age_str = 'YA & OA' if age == 'healthy' else 'YA' if age == 1 else 'OA'
    if semantic:
        rsa_str = 'word2vec'
    else:
        rsa_str = '1st-layer DNN' if early else 'late-layer DNN'
    if short:
        inc_str = ''
    else:
        inc_str = 'Con, Inc, & Neu' if cin is None else \
            'Con' if cin == 1 else 'Inc' if cin == 2 else 'Neu'
    out_str = f'{pre_str} {key_str}. {age_str}. {rsa_str}. {inc_str}'
    return out_str


def get_RSA_fn(inc, age, semantic, DNN_layer, fp_fMRI_col='obj_fMRI',
               pre_str='', PCA_obj=False,
               combine_regions=False, bilateral=False, vec_prod=False,
               org_by_region=False):
    age_str = 'healthy' if age == 'healthy' else \
        'YA' if age == 1 else 'OA'
    inc_str = '' if inc is None else \
        '_Con' if inc == 1 else \
        '_Inc' if inc == 2 else \
        '_Neu' if inc == 3 else 'BAD_CIN'
    sem_str = '_sem' if semantic else ''
    # el_str = '' if semantic else '_early' if early else '_late'
    combine_str = '_comb' if combine_regions else ''
    bilat_str = '_bil' if bilateral else ''
    vecprod_str = '_vecprod' if vec_prod else ''
    by_region_str = '_byR' if org_by_region else ''
    PCA_obj_str = '' if PCA_obj else '_PCAallImg'

    fp_in_str = fp_fMRI_col.replace('fMRI', '')

    dnn_str = '' if semantic else \
        '_late' if DNN_layer == -1 else \
        '_early' if DNN_layer == 2 else \
        f'_dnn{DNN_layer}'

    fp_out = fr'{pre_str}{fp_in_str}{age_str}{inc_str}{sem_str}{dnn_str}' \
             fr'{combine_str}{bilat_str}{vecprod_str}{by_region_str}{PCA_obj_str}'
    print(f'{fp_out=}')
    return fp_out


def prune_to_only_hits(d, key, misses=False):
    d['bhv']['hit_bool'] = np.nan_to_num(d['bhv']['hit_bool'], True).astype(bool)
    for ROI in d['IRAFs_ROI'][key]:
        if d['IRAFs_ROI'][key][ROI].shape[0] != 29:
            mask = np.full(d['IRAFs_ROI'][key][ROI].shape, False)
            # print('bad')
        else:
            # print('good')
            if misses:
                mask = ~d['bhv']['hit_bool']
            else:
                mask = d['bhv']['hit_bool']
        d['IRAFs_ROI'][key][ROI][~mask] = np.nan
        d['activity'][ROI] = np.array(d['activity'][ROI])
        d['activity'][ROI][~mask] = np.nan
    return d


def ndim_tril_flatten(ar):
    tril = np.tril_indices(ar.shape[-1], k=-1)
    slicer = tuple([slice(None)] * (ar.ndim - 2) + [tril[0], tril[1]])
    # print(slicer)
    ar_flat = ar[slicer]
    return ar_flat
