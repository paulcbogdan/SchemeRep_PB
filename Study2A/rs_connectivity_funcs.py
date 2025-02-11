from collections import defaultdict

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from Study1A.modularity_funcs import get_partition_matrix, get_FC_between_ROIs
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
from Study2A.load_Study2A_funcs import load_rs_BOLD
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap


def get_network_partitions():
    from nichord.coord_labeler import get_idx_to_label
    atlas = get_atlas()
    idx_to_label = pickle_wrap(get_idx_to_label, None,
                               kwargs={'coords': atlas['coords'],
                                       'atlas': 'yeo'})
    network2p = defaultdict(list)
    for i, network in idx_to_label.items():
        network2p[network].append(i)
    return network2p


def get_hemi_ps(p_d_ant, p_d_pos, p_v_ant, p_v_pos, schaefer=False,
                combine_regions=False):
    atlas = get_atlas(schaefer=schaefer, combine_regions=combine_regions)
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos, }
    ps_hemi = defaultdict(list)
    for key, p in ps.items():
        # if schaefer:
        #     p = sorted(p, key=lambda x: atlas['coords'][x][1])
        for i in p:
            coord = atlas['coords'][i]
            if coord[0] < 0:
                ps_hemi[f'L{key}'].append(i)
            else:
                ps_hemi[f'R{key}'].append(i)
    return ps_hemi


def get_df_networks(zscore=False, anat_ver=3, add_hemi=True,
                    combine_regions=False,
                    schaefer=False
                    ):
    sn_roi_act, sns, conn_trials = load_rs_BOLD(schaefer=schaefer)
    print('Onto get_df_networks...')

    network2p = pickle_wrap(get_network_partitions)

    key2conn = {}
    for network, p in network2p.items():
        key2conn[network] = get_module_trialwise_z(conn_trials[:, None], p)

    networks = list(key2conn)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                             anat_ver=anat_ver, combine_regions=combine_regions,
                             schaefer=schaefer)
    print(F'{p_v_ant=}')

    # p_v_pos = ([138, 6, 5, 8, 133, 254, 2, 4, 7, 10, 139, 1, 3, 9, 41, 22] +
    #            [23, 20, 16, 18, 14, 21, 13, 17, 12, 142, 15, 19, 136, 140, 135] +
    #            [10, 143, 139, 144, 145, 41, 268, 42, 326, 9, 148, 39] +
    #            [46, 45, 20, 44, 329, 332, 152, 21, 150, 154, 151, 153, 142, 273, 324] +
    #            [375, 24, 1, 38, 376, 28, 131, 3, 369, 35, 25, 27, 4, 42, 40, 325, 371, 22, 34, 26, 370, 9, 39, 338, 23,
    #             36, 44, 31, 20] +
    #            [35, 36, 31, 371, 337, 23, 39, 44, 329, 345, 43, 373, 20, 46, 45, 32, 344, 328, 37, 14, 13, 372, 33, 21,
    #             16, 30, 29, 378])

    # p_v_ant = ([259, 260, 258, 379, 244, 257, 241, 247, 361, 95, 246, 94, 242, 226] +
    #            [252, 248, 250, 249, 385, 251, 399, 102, 366, 104, 90, 386, 265, 367, 263])
    p_v_ant = ([258, 257, 241, 247, 246, 242] +
               [252, 250, 249, 385, 399, 102, 265])


    # p_d_pos = ([184, 185, 183, 158, 113, 314, 384, 269, 88, 309, 160, 159, 138, 313, 383, 311, 308, 112, 315, 271, 161,
    #             87, 312, 8, 50, 310, 162, 126, 6, 89, 270, 147, 272, 109, 139, 93, 146, 10, 268, 143, 145, 41, 144] +
    #            [273, 153, 151, 320, 155, 142, 276, 171, 117, 324, 169, 170, 19, 322, 116, 323, 140, 321, 141, 275, 120,
    #             393, 186, 118, 318, 391, 274, 390, 317, 319, 66, 69, 394, 316, 187, 168, 188, 91, 167, 92, 121, 392])
    #
    # p_d_ant = ([214, 212, 215, 213, 297, 296, 210, 208, 298, 175, 181, 209, 176, 211, 278, 299, 300, 224, 177, 279, 284,
    #             229, 281, 277, 353, 280, 283, 227] +
    #            [235, 305, 356, 217, 292, 303, 359, 289, 218, 288, 179, 290, 398, 395, 178, 223, 304, 397, 222, 396, 220,
    #             216, 221, 219, 182, 301] +
    #            [365, 190, 364, 189, 212, 296, 100, 363, 298, 297, 211, 213, 96, 208, 99] +
    #            [107, 223, 191, 222, 220, 221, 367, 103, 219, 216, 301, 304, 368, 192, 193])

    # print(F'{p_d_ant=}')
    # print(F'{p_d_pos=}')
    print(F'{p_v_ant=}')
    # quit()
    # print(F'{p_v_pos=}')

    if add_hemi:
        ps_hemi = get_hemi_ps(p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                              schaefer=schaefer)
        keys_both = []
        for key, p0 in ps_hemi.items():
            for key1, p1 in ps_hemi.items():
                key_both = f'{key}_{key1}'
                key2conn[key_both] = (
                    get_FC_between_ROIs(conn_trials[:, None],
                                        ps_hemi[key], ps_hemi[key1]))
                keys_both.append(key_both)
        networks += keys_both

    conn_keys, conn_ps = prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos,
                                      p_v_ant, p_v_pos)
    for key, (p0, p1) in zip(conn_keys, conn_ps):
        key2conn[key] = get_FC_between_ROIs(conn_trials[:, None],
                                            p0, p1)

    num_ROIs = get_num_ROIs(schaefer)

    key2conn['FC_all'] = get_module_trialwise_z(conn_trials[:, None],
                                                list(range(num_ROIs)))

    act_keys = ['dp', 'da', 'vp', 'va']
    act_p = [p_d_pos, p_d_ant, p_v_pos, p_v_ant]
    key2p_M = {}
    for key, p in zip(act_keys, act_p):
        key2p_M[key] = np.nanmean(sn_roi_act[:, p, :], axis=1)
    act_keys += 'no'
    p_no = [i for i in range(num_ROIs) if i not in p_dorsal + p_ventral]
    key2p_M['no'] = np.nanmean(sn_roi_act[:, p_no, :], axis=1)

    df_as_d = defaultdict(list)
    n_TRs = sn_roi_act.shape[-1]
    for i, sn in enumerate(sns):
        for key, conn in key2conn.items():
            vals = conn[i, :]
            if zscore: vals = stats.zscore(vals)
            df_as_d[key].extend(vals)
            n_TRs = len(vals)
        for key, M in key2p_M.items():
            vals = M[i, :]
            if zscore: vals = stats.zscore(vals)
            df_as_d[key].extend(vals)

        df_as_d['sn'].extend([sn] * n_TRs)
    df = pd.DataFrame(df_as_d)
    networks += ['dd', 'vv', 'dv_ant', 'dv_pos']
    return df, networks


def partial_corr_df(df, cols, cov, verbose=1):
    cols = [col for col in cols if col not in cov]
    ar = np.full((len(cols), len(cols)), np.nan, dtype=float)
    print(df[cols])
    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i < j:
                try:
                    out = (pg.partial_corr(data=df, x=col_i, y=col_j,
                                           covar=cov).
                           round(3))

                    ar[i, j] = out['r'].values[0]
                    ar[j, i] = ar[i, j]
                except AssertionError as e:
                    ar[i, j] = np.nan
                    ar[j, i] = np.nan

    df_result = pd.DataFrame(ar, index=cols, columns=cols)
    if verbose:
        print(df_result)
    return ar


def get_num_ROIs(schaefer):
    if isinstance(schaefer, tuple):
        num_ROIs = schaefer[1]
    elif isinstance(schaefer, bool):
        if schaefer:
            num_ROIs = 1000
        else:
            num_ROIs = 246
    else:
        raise ValueError
    return num_ROIs


def prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos,
                 schaefer=False):
    num_ROIs = get_num_ROIs(schaefer)

    ad_else = list(set(range(num_ROIs)) - set(p_d_ant))
    pd_else = list(set(range(num_ROIs)) - set(p_d_pos))
    av_else = list(set(range(num_ROIs)) - set(p_v_ant))
    pv_else = list(set(range(num_ROIs)) - set(p_v_pos))

    no_match = list(set(range(num_ROIs)) -
                    set(p_d_ant + p_d_pos + p_v_ant + p_v_pos))

    dd_else = list(set(range(num_ROIs)) - set(p_d_ant + p_d_pos))
    dv_ant_else = list(set(range(num_ROIs)) - set(p_d_ant + p_v_ant))
    vv_else = list(set(range(num_ROIs)) - set(p_v_ant + p_v_pos))
    dv_pos_else = list(set(range(num_ROIs)) - set(p_d_pos + p_v_pos))

    p_ant = list(set(p_d_ant + p_v_ant))
    p_pos = list(set(p_d_pos + p_v_pos))

    # allow diagonal
    pd_no = list(set(range(num_ROIs)) - set(p_d_pos + p_d_ant + p_v_pos))
    ad_no = list(set(range(num_ROIs)) - set(p_d_ant + p_d_pos + p_v_ant))
    pv_no = list(set(range(num_ROIs)) - set(p_v_pos + p_v_ant + p_d_pos))
    av_no = list(set(range(num_ROIs)) - set(p_v_ant + p_v_pos + p_d_ant))

    conn_keys = ['dd', 'vv',
                 'dv_ant', 'dv_pos',
                 'dpva', 'vpda',
                 'pd_else', 'ad_else',
                 'pv_else', 'av_else',
                 'dd_else', 'vv_else',
                 'dv_ant_else', 'dv_pos_else',
                 'pd_no', 'ad_no',
                 'pv_no', 'av_no',
                 'dd_no', 'vv_no',
                 'dv_ant_no', 'dv_pos_no',
                 'pd_no2', 'ad_no2',
                 'pv_no2', 'av_no2',
                 'no_no',

                 'pd_no_L', 'pd_no_R',
                 'ad_no_L', 'ad_no_R',
                 'pv_no_L', 'pv_no_R',
                 'av_no_L', 'av_no_R',
                 ]

    p_d_pos_L = [i for i in p_d_pos if i % 2 == 0]
    p_d_pos_R = [i for i in p_d_pos if i % 2 == 1]
    p_d_ant_L = [i for i in p_d_ant if i % 2 == 0]
    p_d_ant_R = [i for i in p_d_ant if i % 2 == 1]
    p_v_pos_L = [i for i in p_v_pos if i % 2 == 0]
    p_v_pos_R = [i for i in p_v_pos if i % 2 == 1]
    p_v_ant_L = [i for i in p_v_ant if i % 2 == 0]
    p_v_ant_R = [i for i in p_v_ant if i % 2 == 1]
    conn_ps = [(p_d_pos, p_d_ant), (p_v_pos, p_v_ant),
               (p_d_ant, p_v_ant), (p_d_pos, p_v_pos),
               (p_d_pos, p_v_ant), (p_v_pos, p_d_ant),
               (p_d_pos, pd_else), (p_d_ant, ad_else),
               (p_v_pos, pv_else), (p_v_ant, av_else),
               (p_dorsal, dd_else), (p_ventral, vv_else),
               (p_ant, dv_ant_else), (p_pos, dv_pos_else),
               (p_d_pos, no_match), (p_d_ant, no_match),
               (p_v_pos, no_match), (p_v_ant, no_match),
               (p_dorsal, no_match), (p_ventral, no_match),
               (p_ant, no_match), (p_pos, no_match),
               (p_d_pos, pd_no), (p_v_pos, pv_no),
               (p_d_ant, ad_no), (p_v_ant, av_no),
               (no_match, no_match),
               (p_d_pos_L, no_match), (p_d_pos_R, no_match),
               (p_d_ant_L, no_match), (p_d_ant_R, no_match),
               (p_v_pos_L, no_match), (p_v_pos_R, no_match),
               (p_v_ant_L, no_match), (p_v_ant_R, no_match),
               ]
    return conn_keys, conn_ps


def get_module_trialwise_z(sn_inc_conn_trials, p_module):
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1)  # omit inc axis
    return sn_agg_trials_dd
