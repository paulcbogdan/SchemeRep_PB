import os

from Study2A.rs_connectivity_funcs import prep_conn_ps

os.chdir(r'C:\PycharmProjects\SchemeRep')

from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import linalg, stats as stats
from tqdm import tqdm

from Utils.atlas_funcs import get_atlas
from Study1A.modularity_funcs import get_modules, get_partition_matrix, get_FC_between_ROIs
from Utils.plotting_funcs import plot_connectivity
from Utils.pickle_wrap_funcs import pickle_wrap
# from old_Apr6.vendor_lmers import get_module_trialwise_z
# from load_more import load_act_conn
from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions


def high_variance_conn_confounds(conn_trials, tile=.02, n_confounds=5):
    for i in tqdm(range(conn_trials.shape[0]), desc='high variance confounds'):
        sn_conn = conn_trials[i]
        trils = np.tril_indices(sn_conn.shape[1], k=-1)
        sn_flat_T = sn_conn[trils[0], trils[1], :].T
        v_flat = np.nanvar(sn_flat_T, axis=0) # participant 27 has all zero in two trials
        v_flat[np.isnan(v_flat)] = 0
        top_tile_idx = int(v_flat.shape[0] * tile)
        top_idxs = np.argsort(v_flat)[-top_tile_idx:]
        sn_top_T = sn_flat_T[:, top_idxs]
        num_nans = np.sum(np.isnan(sn_top_T))
        try:
            U, S, Vh = np.linalg.svd(sn_top_T)
        except np.linalg.LinAlgError:
            print(f'LinAlgError: {i}')
            plt.imshow(np.isnan(sn_top_T), aspect='auto')
            plt.colorbar()
            plt.title(f'{num_nans=}')
            plt.show()
            continue
        confounds = U[:, :n_confounds]

        # Taken from nilearn signal.clean
        #   https://pages.stat.wisc.edu/~larget/math496/qr.html
        # I believe:
        #   betas = Q.T.dot(sn_top_T)
        #   so subtracting Q.dot(betas) is regressing out the effect
        Q, R, _ = linalg.qr(confounds, mode="economic", pivoting=True)
        Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float64).eps * 100.0]
        sn_flat_T -= Q.dot(Q.T).dot(sn_flat_T)
        # print(sn_flat_T.shape)
        # print(sn_flat_T)
        # quit()

        sn_conn[trils[0], trils[1], :] = sn_flat_T.T
        sn_conn[trils[1], trils[0], :] = sn_flat_T.T

    return conn_trials


def get_key2conn_hemi(p_d_ant, p_d_pos, p_v_ant, p_v_pos):
    atlas = get_atlas()
    ps = {'da': p_d_ant, 'dp': p_d_pos, 'va': p_v_ant, 'vp': p_v_pos,}
    ps_hemi = defaultdict(list)
    for key, p in ps.items():
        for i in p:
            coord = atlas['coords'][i]
            if coord[0] < 0:
                ps_hemi[f'L{key}'].append(i)
            else:
                ps_hemi[f'R{key}'].append(i)
    ps_hemi.update(ps)
    return ps_hemi


def get_interesting_hemi_cols():
    cols_order_dd = ['Ldp_Lda', 'Rdp_Rda', # top = main direction
                     'Ldp_Rdp', 'Lda_Rda', # middle = hemi-bounce
                     'Ldp_Rda', 'Lda_Rdp'] # bottom = cross
    cols_order_vv = [col.replace('d', 'v') for col in cols_order_dd]
    cols_order_p  = ['Ldp_Lvp', 'Rdp_Rvp',
                     'Ldp_Rdp', 'Lvp_Rvp',
                     'Ldp_Rvp', 'Lvp_Rdp']
    cols_order_a  = [col.replace('p', 'a') for col in cols_order_p]

    # cols_order_L = ['Ldp_Lda', 'Lvp_Lva',
    #                 'Ldp_Lvp', 'Lda_Lva',
    #                 'Ldp_Lva', 'Lda_Lvp']
    # cols_order_R = [col.replace('L', 'R') for col in cols_order_L]
    cols_order = []
    cols_order += cols_order_dd + cols_order_vv + cols_order_p + cols_order_a
    # cols_order += cols_order_L + cols_order_R

    return cols_order


def do_modularity_hemi(df):
    cols_analyze = get_interesting_hemi_cols() + ['FC_all']
    cols_analyze = list(set(cols_analyze))
    corr = np.array(df[cols_analyze].corr())
    tick_lows = np.arange(0, len(cols_analyze))
    ticks = tick_lows# + 0.5
    tick_labels = cols_analyze

    corr[np.diag_indices_from(corr)] = np.nan
    plot_connectivity(corr, ticks, tick_labels, tick_lows, no_avg=True, vmin=0, vmax=1.0)

    thr = np.nanquantile(corr, .8)
    corr[corr < thr] = 0
    corr[corr >= thr] = 1
    partitions = get_modules(corr)
    for i, p in enumerate(partitions):
        p_named = [cols_analyze[j] for j in p]
        print(f'{i}: {p=} ({p_named})')

    for p in partitions:
        corr_vp = get_partition_matrix(np.ones(corr.shape), p,
                                        w_zeros=True)
        plot_connectivity(corr_vp, ticks, tick_labels, tick_lows, title='', no_avg=True, vmin=0, vmax=1)


def get_rs_vendor_df(roiwise=False, do_hemi=False, high_var_confounds=False,
                     zscore=True, norm_std=True, f=None, YA_only=False):
    sn_roi_act, sns, conn_trials = load_act_conn(norm_std, f=f,
                                                 YA_only=YA_only)

    # sn_roi_act = stats.rankdata(sn_roi_act, axis=2)
    # conn_trials = np.abs(sn_roi_act[..., None, :] - sn_roi_act[..., None, :, :])

    # sn_roi_act = np.random.normal(size=sn_roi_act.shape)
    if high_var_confounds:
        fp_pkl = rf'cache/high_var_conn_trials.pkl'
        conn_trials = pickle_wrap(lambda: high_variance_conn_confounds(conn_trials),
                                  fp_pkl, easy_override=True)
    conn_trials = conn_trials[:, None, :, :, :]
    diag = np.diag_indices(conn_trials.shape[3])
    conn_trials[:, 0, diag[0], diag[1], :] = np.nan

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=True, scrub=False)

    p_d_ant = p_d_ant[:32]  # Making all equal length
    p_d_pos = p_d_pos[:32]
    p_v_ant = p_v_ant[:32]
    p_v_pos = p_v_pos[:32]

    conn_keys, conn_ps = prep_conn_ps(p_dorsal, p_ventral, p_d_ant, p_d_pos,
                                      p_v_ant, p_v_pos)
    key2conn = {}
    for key, (p0, p1) in zip(conn_keys, conn_ps):
        key2conn[key] = get_FC_between_ROIs(conn_trials, p0, p1)

    act_keys = ['dp', 'da', 'vp', 'va']
    act_p = [p_d_pos, p_d_ant, p_v_pos, p_v_ant]
    key2p_M = {}
    for key, p in zip(act_keys, act_p):
        key2p_M[key] = np.nanmean(sn_roi_act[:, p, :], axis=1)

    if do_hemi:
        ps_hemi = get_key2conn_hemi(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
        key2conn_hemi = {}
        for i, p0 in enumerate(ps_hemi):
            for j, p1 in enumerate(ps_hemi):
                key = f'{p0}_{p1}'
                key2conn_hemi[key] = get_FC_between_ROIs(conn_trials,
                                                         ps_hemi[p0],
                                                         ps_hemi[p1])
    key2conn['FC_all'] = get_module_trialwise_z(conn_trials, list(range(246)))

    if roiwise:
        quads = ['dp', 'da', 'vp', 'va']
        quad2p = {'dp': p_d_pos, 'da': p_d_ant, 'vp': p_v_pos, 'va': p_v_ant}
        for i in range(246):
            for quad in quads:
                key2conn[f'{quad}_{i}'] = \
                    get_FC_between_ROIs(conn_trials, [i],
                                        quad2p[quad])
            key2p_M[f'M_{i}'] = sn_roi_act[:, i, :]


    df_as_d = defaultdict(list)
    n_TRs = sn_roi_act.shape[-1]
    for i, sn in enumerate(sns):
        for key, conn in key2conn.items():
            vals = conn[i, :]
            if zscore:
                vals = stats.zscore(vals)
            df_as_d[key].extend(vals)
        if do_hemi:
            for key, conn in key2conn_hemi.items():
                vals = conn[i, :]
                if zscore: vals = stats.zscore(vals)
                df_as_d[key].extend(vals)
        for key, M in key2p_M.items():
            vals = M[i, :]
            if sn == '138':
                print(f'{key=}, {vals=}')
            if zscore: vals = stats.zscore(vals)

            df_as_d[key].extend(vals)

        df_as_d['sn'].extend([sn] * n_TRs)
    df = pd.DataFrame(df_as_d)
    return df, conn_keys

def generate_Figure_5A_correlation():
    df, conn_keys = pickle_wrap(get_rs_vendor_df,
                                kwargs={'roiwise': False,
                                        'do_hemi': False,
                                        'zscore': False,
                                'high_var_confounds': False},
                                easy_override=False)

    hemis = get_interesting_hemi_cols()

    # keys = act_keys + conn_keys
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df['FC'] = df[conn_keys].sum(axis=1)
    # print(df['dFC'].describe())
    # quit()

    # print(df[act_keys + conn_keys + ['FC_all']].corr())
    # quit()

    df['age'] = df['sn'].apply(lambda sn: int(str(sn)[0]))
    df['horz'] = df['dd'] + df['vv']
    df['vert'] = df['dv_ant'] + df['dv_pos']
    df['age'] = stats.zscore(df['age'], nan_policy='omit') #
    # formula = ('dd ~ vv + dv_ant + dv_pos + ' # dpva + vpda +
    #            'FC_all + pd_else + ad_else + ' #  dd_else +
    #            'dp + da + vp + va + '
    #            '(1 + vv + dv_ant + dv_pos | sn)')

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    print(df)
    quit()

    # formula = ('dd_vv ~ dv_dv + ' # dpva + vpda +
    #            'FC_all + ' #  dd_else + # pd_else + ad_else +
    #            'dp + da + vp + va + '
    #            '(1 + vv + dv_ant + dv_pos | sn)')

    # drop = {'Ldp_Lda', 'Rdp_Rda', 'Lda_Ldp', 'Rda_Rdp',
    #         'Ldp_Rda', 'Rdp_Lda', 'Rda_Ldp', 'Lda_Rdp', }
    # hemis_str = '+'.join([hemi for hemi in hemis if hemi not in drop])
    # formula = ('dd ~ ' + hemis_str + ' + '
    #            'FC_all + '
    #            '(1 | sn)')

    # pd_else + ad_else +
    #                'pd_no + ad_no + pv_no + av_no +' # pv_else + av_else +

    formula = ('vv ~ dv_ant + dv_pos + dd + ' # dpva + vpda +  dpva + vpda + 
               'pd_no + ad_no + pv_else + av_else + ' # pv_else + av_else +  + pv_else + av_else +
               'FC_all + ' #  FC_all + 
               'dp + da + vp + va + '
               '(1 + dv_ant + dv_pos | sn)') # + vv + dv_ant + dv_pos

    from pymer4 import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

if __name__ == '__main__':
    generate_Figure_5A_correlation()