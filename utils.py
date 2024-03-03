from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
from scipy import stats as stats

from datetime import datetime
from time import time
from pickle import UnpicklingError
import pandas as pd
from nilearn import image
import inspect
import functools
from functools import cache
from colorama import Fore

def regress_out(x, y):
    x = np.array(x)
    y = np.array(y)
    nans = np.isnan(x) | np.isnan(y)
    n_goods = np.sum(~nans)
    if n_goods < 3:
        print(f'Warning: not enough non-nan values to regress out: {n_goods=}')
        return y
    b, m, r, p, er = stats.linregress(x[~nans], y[~nans])
    return y - x*b

def regress_out_multi(X, y):
    for x in X:
        y = regress_out(x, y)
    return y

def stdize(v, axis=None, nans=False):
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                            message='invalid value encountered in')

    m = np.nanmean if nans else np.mean
    s = np.nanstd if nans else np.std
    return (v - m(v, axis=axis)[..., None]) / s(v, axis=axis)[..., None]

    # if axis == 3:
    #     return (v - m(v, axis=axis)[:, :, :, None]) / \
    #         s(v, axis=axis)[:, :, :, None]
    # elif axis == 2:
    #     return (v - m(v, axis=axis)[:, :, None]) / s(v, axis=axis)[:, :, None]
    # elif axis == 1:
    #     return (v - m(v, axis=axis)[:, None]) / s(v, axis=axis)[:, None]
    # else:
    #     return (v - m(v, axis=axis)) / s(v, axis=axis)


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

def pb_outer_euc(a, b, flat=False, tril=False, nan_diag=False):
    a = np.array(a)
    b = np.array(b)
    a = np.expand_dims(a, axis=2)
    b = np.expand_dims(b, axis=1)
    c = -abs(a - b)
    if nan_diag:
        assert c.shape[-1] == c.shape[-2], f'c must be square: {c.shape=}'
        c[:, np.arange(c.shape[-1]), np.arange(c.shape[-1])] = np.nan
    if tril:
        c = tril_flat(c)
    elif flat:
        c = c.reshape(-1, c.shape[1]*c.shape[2])
    return c

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
        c = tril_flat(c)
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
        key_str = 'Object stim,'
    elif key == 'scn':
        key_str = 'Scene stim,'
    elif key == 'scn_':
        key_str = 'Scene stim (regressed obj. effect)\n'
    elif key == 'scn_abs':
        key_str = 'Abs-scene,'
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
    # if key != 'scn_':
    #     key_str += '.'
    return key_str

def make_title_str(pre_str, key, age, DNN_layer, semantic,
                   fp=None,
                   cin=None, short=False):
    age_str = 'YA & OA' if age == 'healthy' else 'YA' if age == 1 else \
        'OA' if age == 2 else age
    if semantic:
        rsa_str = 'word2vec RSA'
    else:
        if DNN_layer == 2:
            rsa_str = '1st-layer DNN RSA'
        elif DNN_layer == -1:
            rsa_str = 'last-layer DNN RSA'
    if short:
        inc_str = ''
    else:
        inc_str = 'Con, Inc, & Neu' if cin is None else \
            'Con' if cin == 1 else 'Inc' if cin == 2 else \
                'Neu' if cin == 3 else cin

    if fp:
        fp2str = {'obj4_fMRI': 'Object betas',
                  'scn4_fMRI': 'Scene betas',
                  'obj7_fMRI': 'Object betas',
                  'scn7_fMRI': 'Scene betas',
                  'con7_fMRI': 'Concept ret. betas',
                  'vis7_fMRI': 'Visual ret. betas',}
        fp_str = fp2str[fp]
    else:
        fp_str = ''
    stim_str = get_RSA_name(key)

    # print(f'{fp=}')
    # print(f'{fp_str=}')
    # quit()

    out_str = f'{pre_str} {fp_str}, {stim_str} {rsa_str}.\n{age_str}. {inc_str}'
    return out_str

def load_ni_w_nan_fps(fps):
    nan_idxs = pd.isna(fps)
    good_idxs = ~nan_idxs
    if nan_idxs.any():
        img = image.load_img(fps[good_idxs]).get_fdata()
        # print(img.shape)
        # quit()
        blank = np.full((img.shape[0], img.shape[1], img.shape[2], len(fps)),
                        np.nan)
        blank[:, :, :, good_idxs] = img
        img = blank
    else:
        img = image.load_img(fps).get_fdata()
    # num_goods = np.sum(~nan_idxs)
    return img, good_idxs

def get_RSA_fn(inc, age, semantic, DNN_layer, fp_fMRI_col='obj_fMRI',
               pre_str='', PCA_obj=True,
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


def tril_flat(ar):
    tril = np.tril_indices(ar.shape[-1], k=-1)
    # slicer = tuple([slice(None)] * (ar.ndim - 2) + [tril[0], tril[1]])
    # print(slicer)
    # ar_flat = ar[slicer]
    ar_flat = ar[..., tril[0], tril[1]]
    return ar_flat


import pickle
import os
from time import time

def getVariableName(variable, globalVariables):
    # from: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    """ Get Variable Name as String by comparing its ID to globals() Variables' IDs
        args:
            variable(var): Variable to find name for (Obviously this variable has to exist)
        kwargs:
            globalVariables(dict): Copy of the globals() dict (Adding to Kwargs allows this function to work properly when imported from another .py)
    """
    for globalVariable in globalVariables:
        if id(variable) == id(globalVariables[globalVariable]): # If our Variable's ID matches this Global Variable's ID...
            return globalVariable # Return its name from the Globals() dict

def obj2str(val):
    kwargs_str = ''
    if isinstance(val, np.ndarray):
        return ''
    elif isinstance(val, dict):
        # val_d = ''
        for key2 in sorted(val.keys()):
            kwargs_str += obj2str(val[key2])
            # val_d += f'{val[key2]}_'
        # kwargs_str += f'{val_d}_'
    elif isinstance(val, list):
        # val_l = ''
        for val_ in val:
            kwargs_str += obj2str(val_)
            # val_l += f'{val_}_'
        # kwargs_str += f'{val_l}_'
    else:
        kwargs_str += f'{val}_'
    if len(kwargs_str) > 20:
        target_len = 10
        skipper = len(kwargs_str) // target_len
        kwargs_str = kwargs_str[::skipper]

    return kwargs_str

def f2str(callback, kwargs=None):
    if kwargs is None:
        kwargs = {}
    signature = inspect.signature(callback) # functools.partial impacts sig
    for k, v in signature.parameters.items():
        if v.default is v.empty: continue # exclude args, only want kwargs
        if k not in kwargs: kwargs[k] = v.default
    if kwargs is not None:
        kwargs_str = ''
        for key in sorted(kwargs.keys()):
            val = kwargs[key]
            kwargs_str += obj2str(val)
        kwargs_str = kwargs_str[:-1]
    return kwargs_str

def get_default_fp(args, kwargs, callback, cache_dir, verbose=0):
    if verbose: print(f'Making default filepath: {kwargs=}')
    if args is not None:
        args_str = '_'.join(args)
    else:
        args_str = ''
    kwargs_str = f2str(callback, kwargs)
    if isinstance(callback, functools.partial):
        name = callback.func.__name__
    else:
        name = callback.__name__
    func_dir = f'{cache_dir}/{name}'
    Path(func_dir).mkdir(parents=True, exist_ok=True)
    filepath = f'{func_dir}/{args_str}_{kwargs_str}.pkl'
    # Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # filepath = f'{cache_dir}/{name}_{args_str}_{kwargs_str}.pkl'
    if verbose: print(f'Default pickle_wrap filepath: {filepath}')



    # bad_chars = [';', ':', '!', "*", " ", '{', '}', ',', '\'']
    # for char in bad_chars:
    #     filepath = filepath.replace(char, '')
    # print(f'Default pickle_wrap filepath: {filepath}')
    # quit()
    return filepath

PICKLE_CACHE = {}

def pickle_wrap(callback, filepath=None, args=None, kwargs=None,
                easy_override=False, verbose=0, cache_dir='cache',
                dt_max=None, RAM_cache=False):
    '''
    :param filepath: File to which the callback output should be loaded (if already created)
                     or where the callback output should be saved
    :param callback: Function to be pickle_wrapped
    :param args: Arguments passed to function (often not necessary)
    :param kwargs: Kwargs passed to the function (often not necessary)
    :param easy_override: If true, then the callback will be performed and the previous .pkl
                          save will be overwritten (if it exists)
    :param verbose: If true, then some additional details will be printed (the name of the callback,
                    and the time needed to perform the function or load the .pkl)
    :return: Returns the output of the callback or the output saved in filepath
    '''
    if filepath is None:
        filepath = get_default_fp(args, kwargs, callback, cache_dir, verbose)
    if RAM_cache and filepath in PICKLE_CACHE:
        return PICKLE_CACHE[filepath]

    if verbose:
        print(f'pickle_wrap: {filepath=}')
        print('\tFunction:', getVariableName(callback,
                                             globalVariables=globals().copy()))

    if os.path.isfile(filepath):
        made = os.path.getmtime(filepath)
        fn = os.path.basename(filepath)
        if dt_max is None:
            dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)
        dt_min = datetime(2023, 11, 1, 14, 0, 0, 0)

        dt = datetime.fromtimestamp(made)
        if dt < dt_max and dt > dt_min:
            easy_override = True
            print(f'File ({fn}) was made: {made}')
            print('\tFile is old, overriding')

    if os.path.isfile(filepath) and not easy_override:
        try:
            start = time()
            with open(filepath, "rb") as file:
                pk = pickle.load(file)
                if verbose: print(f'\tLoad time: {time()-start:.3f} s')
                if not verbose: print(f'Pickle loaded: {filepath=} '
                                      f'({time() - start:.3f} s)')
                if RAM_cache: PICKLE_CACHE[filepath] = pk
                return pk
        except (UnpicklingError, MemoryError) as e:
            print(f'{Fore.RED}{e=}')
            print(f'\t{Fore.YELLOW}{callback=}')
            print(f'\t{Fore.YELLOW}{filepath=}{Fore.RESET}')


    if verbose:
        print('Callback:',
              getVariableName(callback, globalVariables=globals().copy()))
    start = time()
    if args:
        output = callback(*args)
    elif kwargs:
        output = callback(**kwargs)
    else:
        output = callback()
    if verbose:
        print(f'\tFunction time: {time()-start:.3f} s')
        # print('\tDumping to file name:', filepath)
    start = time()
    try:
        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
        if not verbose: print(f'Pickle wrapped: {filepath=} '
                              f'({time()-start:.3f} s)')
    except FileNotFoundError:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
    if verbose: print(f'\tDump time: {time()-start:.3f} s')
    if RAM_cache: PICKLE_CACHE[filepath] = output
    return output

def timing(f):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


def get_formula_cols(df, formula):
    import re
    formula = re.split(' |[*]|\)|\(', formula)
    cols = []
    for col in df.columns:
        # if col == 'sn': continue
        if col in formula:
            cols.append(col)
    return cols

if __name__ == '__main__':
    # f = lambda: pickle_wrap(1, None)
    f = pickle_wrap
    signature = inspect.signature(f)
    # print(signature)
    for k, v in signature.parameters.items():
        # print(isinstance(v, ))
        print(f'{k}: {v.default=}')
        # print(v.default is v.empty)


def run_two_sample_on_2D(ar0, ar1):
    YA_M = np.nanmean(ar0, axis=0)
    OA_M = np.nanmean(ar1, axis=0)
    YA_sd = np.nanstd(ar0, axis=0)
    OA_sd = np.nanstd(ar1, axis=0)
    YA_N = np.sum(~np.isnan(ar0), axis=0)
    OA_N = np.sum(~np.isnan(ar1), axis=0)

    both_sd = ((YA_sd ** 2) * (YA_N - 1) +
               (OA_sd ** 2) * (OA_N - 1)) / \
              (YA_N + OA_N - 2)
    both_se = np.sqrt(both_sd * (1 / YA_N + 1 / OA_N))
    t = (YA_M - OA_M) / both_se
    t[np.isnan(t)] = 0
    return t


# HCP_ROOT = r'F:\HCP_Preprocessing\HCP\HCP_WM_data'
HCP_ROOT = r'E:\PycharmProjects_E\HCP_WM'
HCP_CACHE = r'E:\PycharmProjects_E\SchemeRep\cache\HCP_nii'
HCP_RS_ROOT = r'E:\HCP_RS'
