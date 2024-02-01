import os

from network_IRAF import get_formula_cols

os.chdir('C:\PycharmProjects_C\SchemeRep')

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from modularity import get_partition_matrix, get_partition_cross
from network_funcs import load_FC_for_Lifu
from plot_3way_bar import get_vendor_partitions, anterior_posterior_split
from utils import pickle_wrap, stdize, get_RSA_fn
from functools import cache
import pickle
from collections import defaultdict
from functools import wraps
from time import time

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


def prep_IRAF_df_age(age=2, semantic=True, inc=None, bilateral=False,
                     combine_regions=True, vec_prod=False, PCA_obj=True,
                     org_by_region=False, DNN_layer=2, fp_fMRI_col='obj7_fMRI',
                     fp=None, key='scn'):
    if fp is None:
        fn = get_RSA_fn(inc=inc, age=age, semantic=semantic,
                        DNN_layer=DNN_layer, fp_fMRI_col=fp_fMRI_col,
                        PCA_obj=PCA_obj, bilateral=bilateral,
                        combine_regions=combine_regions, vec_prod=vec_prod,
                        org_by_region=org_by_region,
                        )
        fp = fr'cache/RSA/{fn}.pkl'

    with open(fp, 'rb') as file:
        d = pickle.load(file)
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      combine_bilateral=bilateral or org_by_region)

    df_as_d = defaultdict(list)
    n_trials = d['IRAFs_ROI'][key]['IPL_L'].shape[-1]
    # print(d['bhv']['obj'].shape)
    # print(list(d['bhv']))
    # quit()
    sns = np.repeat(np.array(d['sns'])[:, None], n_trials, axis=1)
    sns = np.reshape(sns, -1)
    df_as_d['sn'] = sns
    df_as_d['obj'] = d['bhv']['obj'].reshape(-1)

    df = pd.DataFrame(df_as_d)
    all_regions = []
    ROI_cols = []
    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        IRAFs = d['IRAFs_ROI'][key][ROI]
        IRAFs = np.reshape(IRAFs, -1)
        df[ROI] = IRAFs
        if f'{region}_R' in df.columns:
            df[region] = df[f'{region}_L'] + df[f'{region}_R']
            all_regions.append(region)
    ROI_cols = atlas['ROIs'] + all_regions
    return df, ROI_cols

@timing
def get_trialwise_RSA(semantic=True, inc=None, bilateral=False,
                      combine_regions=True, vec_prod=False, PCA_obj=True,
                      org_by_region=False, DNN_layer=2, fp_fMRI_col='obj7_fMRI',
                      fp=None, key='scn'):
    df_l = []
    for age in [1, 2]:
        df, ROIs = prep_IRAF_df_age(age=age, semantic=semantic, inc=inc,
                                    bilateral=bilateral,
                                    combine_regions=combine_regions,
                                    vec_prod=vec_prod, PCA_obj=PCA_obj,
                                    org_by_region=org_by_region,
                                    DNN_layer=DNN_layer,
                                    fp_fMRI_col=fp_fMRI_col, fp=fp, key=key)
        df['age'] = age
        df_l.append(df)
    df = pd.concat(df_l)
    return df, ROIs


def get_module_trialwise_z(sn_inc_activity_std, p_module):
    sn_inc_conn_trials = sn_inc_activity_std[..., None, :] * \
                         sn_inc_activity_std[..., None, :, :]
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1) # omit inc axis
    return sn_agg_trials_dd

def get_module_cross_trialwise_z(sn_inc_activity_std, p_mod0, p_mod1):
    conn_trials = sn_inc_activity_std[..., None, :] * \
                         sn_inc_activity_std[..., None, :, :]
    conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)
    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    agg_cross = np.nanmean(flat_cross, axis=-1)
    agg_cross = np.nanmean(agg_cross, axis=1) # omit inc axis
    return agg_cross

@timing
@cache
def get_trialwise_vendor(fp='obj7_fMRI', thr=2.0):
    kwargs = {'fp': fp,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'get_df_sn': True,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos = \
        get_vendor_partitions(sn_inc_conn, age2idxs)

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)

    # sn_agg_trials_dd = get_module_trialwise_z(sn_inc_activity_std, p_dorsal)
    # sn_agg_trials_vv = get_module_trialwise_z(sn_inc_activity_std, p_ventral)
    sn_agg_trials_dd = get_module_cross_trialwise_z(sn_inc_activity_std,
                                                        p_d_pos, p_d_ant)
    sn_agg_trials_vv = get_module_cross_trialwise_z(sn_inc_activity_std,
                                                        p_v_pos, p_v_ant)

    sn_agg_trials_dv_ant = get_module_cross_trialwise_z(sn_inc_activity_std,
                                                        p_d_ant, p_v_ant)
    sn_agg_trials_dv_pos = get_module_cross_trialwise_z(sn_inc_activity_std,
                                                        p_d_pos, p_v_pos)
    for i, df_sn in enumerate(df_sns_l):
        df_sn['dd'] = sn_agg_trials_dd[i, :]
        df_sn['vv'] = sn_agg_trials_vv[i, :]
        df_sn['dv_ant'] = sn_agg_trials_dv_ant[i, :]
        df_sn['dv_pos'] = sn_agg_trials_dv_pos[i, :]
        df_sn['sep'] = df_sn['dd'] + df_sn['vv'] #- \
                       # df_sn['dv_ant'] - df_sn['dv_pos']
        df_sn['cross'] = df_sn['dv_ant'] + df_sn['dv_pos']
    df_sns = pd.concat(df_sns_l)
    return df_sns



def do_RSA_x_vendor():
    df_RSA, ROIs = get_trialwise_RSA(fp_fMRI_col='obj7_fMRI', key='scn',
                                     semantic=True)

    df_RSA.set_index(['sn', 'obj'], inplace=True)
    df_vdr = get_trialwise_vendor()
    df_vdr.set_index(['sn', 'obj'], inplace=True)
    df = df_RSA.join(df_vdr)
    df.reset_index(inplace=True, drop=False)

    # df = df[df['age'] == 2]

    pd.set_option('display.precision', 3)

    from pymer4 import Lmer
    for ROI in ROIs:
        formula = f'sep ~ 1 + inc + {ROI} + (1 | sn)'
        cols = get_formula_cols(df, formula)
        df_vals = df[cols].dropna()
        model = Lmer(formula, data=df_vals)
        model.fit(REML=True, verbose=False, summary=False)
        summary = model.coefs
        p = summary['P-val'].loc[ROI]
        print(f'{ROI}: {p=:.4f}')
        if p < .05:
            print(model.summary())
            print('-'*100)

    quit()

if __name__ == '__main__':
    do_RSA_x_vendor()
