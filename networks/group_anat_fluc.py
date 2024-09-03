import copy

import pandas as pd

from atlas_utils import get_atlas
from load_more import load_a
from old.modularity import get_main_partitions
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity
from old_Apr6.fluctuations import partial_corr_df
from utils import pickle_wrap, stdize
from vendor_partitioning import do_regression, get_vendor_partitions
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from scipy import stats, spatial
from functools import cache
from numba import njit, config, jit
from time import time
from data_driven_fluc import load_rs
from time import time
from tqdm import tqdm
from pingouin import partial_corr
from random import random

# suppress RuntimeWarning
from warnings import simplefilter
simplefilter("ignore", category=RuntimeWarning)

COMBINE_REGIONS = False



def get_group_level_d(pda_i, pdp_i, pva_i, pvp_i, p_no):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': COMBINE_REGIONS,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_roi_act, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=-1, cache_dir='cache',
                    RAM_cache=True)

    pdp_i = list(pdp_i)
    pda_i = list(pda_i)
    pvp_i = list(pvp_i)
    pva_i = list(pva_i)
    p_no = list(p_no)

    # sn_inc_conn = []

    # print(pdp_i)
    # print(pda_i)
    # print(sn_inc_conn.shape)
    # print(sn_inc_conn[..., pdp_i].shape)
    inc_dd = sn_inc_conn[:, 0, pda_i, pdp_i]
    inc_dd = np.nanmean(inc_dd, axis=(1, 2))
    # inc_vv = inc_conn[0, *np.ix_(pva_i, pvp_i)]
    inc_vv = sn_inc_conn[:, 0, pva_i, pvp_i]
    inc_vv = np.nanmean(inc_vv, axis=(1, 2))
    # inc_dv_ant = inc_conn[0, *np.ix_(pda_i, pva_i)]
    inc_dv_ant = sn_inc_conn[:, 0, pda_i, pva_i]
    inc_dv_ant = np.nanmean(inc_dv_ant, axis=(1, 2))
    # inc_dv_pos = inc_conn[0, *np.ix_(pdp_i, _pvp_i)]
    inc_dv_pos = sn_inc_conn[:, 0, pdp_i, pvp_i]
    inc_dv_pos = np.nanmean(inc_dv_pos, axis=(1, 2))

    # conn_dd = inc_conn[2, *np.ix_(pda_i, pdp_i)]
    conn_dd = sn_inc_conn[:, 2, pda_i, pdp_i]
    conn_dd = np.nanmean(conn_dd, axis=(1, 2))
    # conn_vv = inc_conn[2, *np.ix_(pva_i, pvp_i)]
    conn_vv = sn_inc_conn[:, 2, pva_i, pvp_i]
    conn_vv = np.nanmean(conn_vv, axis=(1, 2))
    # conn_dv_ant = inc_conn[2, *np.ix_(pda_i, pva_i)]
    conn_dv_ant = sn_inc_conn[:, 2, pda_i, pva_i]
    conn_dv_ant = np.nanmean(conn_dv_ant, axis=(1, 2))
    # conn_dv_pos = inc_conn[2, *np.ix_(pdp_i, pvp_i)]
    conn_dv_pos = sn_inc_conn[:, 2, pdp_i, pvp_i]
    conn_dv_pos = np.nanmean(conn_dv_pos, axis=(1, 2))

    conn_vendor = conn_dd + conn_vv - conn_dv_ant - conn_dv_pos
    inc_vendor = inc_dd + inc_vv - inc_dv_ant - inc_dv_pos
    ef = conn_vendor - inc_vendor
    print(ef.shape)
    quit()

def do_group_anat_fluc():
    conn_trials, sns = load_rs(combine_regions=COMBINE_REGIONS)

    assert all(sns[i] <= sns[i+1] for i in range(len(sns) - 1))
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': False,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_roi_act, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=-1, cache_dir='cache',
                    RAM_cache=True)
    sns_task = [df_sn['sn'].iloc[0] for df_sn in df_sns]
    assert all(sns_task[i] <= sns_task[i+1] for i in range(len(sns_task) - 1))
    bool_overlap = [sn in sns for sn in sns_task]
    sn_inc_conn = sn_inc_conn[bool_overlap]

get_group_level_d((1, 2), (3, 4), (5, 6), (7, 8), (9, 10, 11))
quit()



if __name__ == '__main__':
    do_group_anat_fluc()

