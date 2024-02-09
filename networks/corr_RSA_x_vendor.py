import os

from network_IRAF import get_formula_cols
from activity_ERS import get_df_ERS, get_df_trialwise_MVPA
from old.network_funcs import load_FC_for_Lifu
from ven_x_dor import get_vendor_df

# from old.networks import get_trialwise_vendor

os.chdir('C:\PycharmProjects_C\SchemeRep')

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from old.modularity import get_partition_matrix
# from networks.old import load_FC_for_Lifu
from vendor_partitioning import get_vendor_partitions
from utils import pickle_wrap, stdize, get_RSA_fn, timing
import pickle
from collections import defaultdict
import scipy.stats as stats


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
    rois = d['IRAFs_ROI'][key].keys()
    n_trials = d['IRAFs_ROI'][key][next(iter(rois))].shape[-1]
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
        ROI = ROI.replace(' ', '_')
        if not combine_regions:
            ROI = '_'.join(ROI.split('_')[1:])
        df[ROI] = IRAFs
        ROI_cols.append(ROI)
        if f'{region}_R' in df.columns:
            df[region] = df[f'{region}_L'] + df[f'{region}_R']
            # ROI_cols.append(region)
            all_regions.append(region)
    ROI_cols += all_regions
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


def get_module_trialwise_z(sn_inc_conn_trials, p_module):
    # sn_inc_conn_trials = sn_inc_activity_std[..., None, :] * \
    #                      sn_inc_activity_std[..., None, :, :]
    sn_inc_conn_trials = np.transpose(sn_inc_conn_trials, (0, 1, 4, 2, 3))
    sn_inc_conn_trials_dd = get_partition_matrix(sn_inc_conn_trials, p_module)
    tridx_dd = np.tril_indices(sn_inc_conn_trials_dd.shape[-1], k=-1)
    sn_inc_flat_trails_dd = sn_inc_conn_trials_dd[:, :, :,
                            tridx_dd[0], tridx_dd[1]]
    sn_inc_agg_trials_dd = np.nanmean(sn_inc_flat_trails_dd, axis=-1)
    sn_agg_trials_dd = np.nanmean(sn_inc_agg_trials_dd, axis=1) # omit inc axis
    return sn_agg_trials_dd


@timing
def get_trialwise_ss_vendor(group_exemplar=False, memory=False):
    if memory:
        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'hit_hit',
                  'atlas_name': 'BNA',
                  'key_vals': (False, False, True),
                  'get_df_sn': True,
                  }
    else:
        kwargs = {'fp': 'obj7_fMRI',
                  'key': 'inc',
                  'atlas_name': 'BNA',
                  'key_vals': (1, 2, 3),
                  'get_df_sn': True,
                  }

    # TODO: could swap this for a subsequent memory effect?
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns_l = \
        pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                    easy_override=False, cache_dir='cache')
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(sn_inc_conn, age2idxs)

    sn_inc_activity_std = stdize(sn_inc_activity, axis=3, nans=True)
    p_dorsal_ = np.zeros(sn_inc_activity_std.shape[2], dtype=bool)
    p_dorsal_[p_dorsal] = True
    sn_inc_activity_std[:, :, ~p_dorsal_, :] = np.nan

    z_trials = sn_inc_activity_std[..., None, :] * \
               sn_inc_activity_std[..., None, :, :]
    # z_trials = np.repeat(matrix_mask[None, None, ..., None],
    #                      z_trials.shape[-1], axis=4) * z_trials[..., :]

    trils = np.tril_indices(z_trials.shape[-2], k=-1)
    z_flat_trials = z_trials[:, :, trils[0], trils[1], :]
    z_flat_trials = np.nanmean(z_flat_trials, axis=1)
    z_flat_trials_broad = z_flat_trials[:, None, :, :]
    z_flat_trials_broad_std = stdize(z_flat_trials_broad, axis=2, nans=True)

    cond_conn = np.nanmean(z_trials, axis=-1)
    cond_flat = cond_conn[:, :, trils[0], trils[1]]
    if group_exemplar:
        cond_flat = np.nanmean(cond_flat, axis=0)[None, :, :]
    # print(cond_flat.shape)
    # quit()
    cond_flat -= np.nanmean(cond_flat, axis=1)[:, None]

    cond_flat_std = stdize(cond_flat, axis=2, nans=True)
    corr_trials = z_flat_trials_broad_std * cond_flat_std[:, :, :, None]
    corr_trials = np.nanmean(corr_trials, axis=2)
    prev_cols = set(df_sns_l[0].columns)
    for i, df_sn in enumerate(df_sns_l):
        df_sn['smlr_i'] = corr_trials[i, 0, :]
        df_sn['smlr_n'] = corr_trials[i, 1, :]
        df_sn['smlr_c'] = corr_trials[i, 2, :]
        df_sn['smlr_ci'] = df_sn['smlr_c'] - df_sn['smlr_i']
    new_cols = set(df_sns_l[0].columns) - prev_cols
    df_sns = pd.concat(df_sns_l)
    return df_sns, new_cols




def do_RSA_x_vendor():
    # kwargs = {
    #     'fp_fMRI_col': 'scn7_fMRI',
    #     'key': 'scn',
    #     'semantic': True,
    #     'combine_regions': False
    # }
    kwargs = {
        'fp_fMRI_col': 'scn7_fMRI',
        'key': 'scn',
        'semantic': True,
        'combine_regions': True
    }

    # kwargs = {
    #     'fp_fMRI_col': 'obj7_fMRI',
    #     'key': 'obj',
    #     'semantic': False,
    #     'combine_regions': True
    # }
    df_RSA, ROIs = pickle_wrap(None, get_trialwise_RSA, kwargs=kwargs,
                               cache_dir='cache', easy_override=False)
    df_RSA.set_index(['sn', 'obj'], inplace=True)

    df_ERS, ERS_ROIs, _ = pickle_wrap(None, get_df_ERS, kwargs={},
                               cache_dir='cache', easy_override=True)
    df_ERS.set_index(['sn', 'obj'], inplace=True)

    df_ERS_obj, ERS_obj_ROIs, _ = pickle_wrap(None, get_df_ERS,
                                           kwargs={'fp0': 'obj7_fMRI'},
                               cache_dir='cache', easy_override=False)
    df_ERS_obj.set_index(['sn', 'obj'], inplace=True)
    df_ERS = df_ERS.join(df_ERS_obj[ERS_obj_ROIs])


    df_RSA = df_RSA.join(df_ERS[ERS_ROIs + ERS_obj_ROIs])

    df_ss_vdr, new_cols = pickle_wrap(None, get_trialwise_ss_vendor,
                                      kwargs={'group_exemplar': False},
                         cache_dir='cache', easy_override=False)

    df_ss_vdr.set_index(['sn', 'obj'], inplace=True)
    df_ss_vdr = df_ss_vdr[new_cols]

    df_vdr = pickle_wrap(None, get_vendor_df, kwargs={'fp': 'obj7_fMRI'},
                         cache_dir='cache', easy_override=False)

    df_vdr.set_index(['sn', 'obj'], inplace=True)
    df = df_RSA.join(df_vdr, rsuffix='meh')
    df = df.join(df_ss_vdr)

    df_MVPA, ROIs_mvpa, _, _ = get_df_trialwise_MVPA('obj7_fMRI', key='inc',
                                                  vals=(1, 3),
                                                  combine_regions=True)

    df_MVPA.set_index(['sn', 'obj'], inplace=True)
    df = df.join(df_MVPA[ROIs_mvpa])
    df.reset_index(inplace=True, drop=False)
    # df = df[df['age'] == 2]
    # df = df[df['inc'] == 3]

    from pymer4 import Lmer

    # formula_mem = 'vendor ~ 1 + smlr_ci + (1 | sn)'
    # model = Lmer(formula_mem, data=df)
    # model.fit(REML=True, verbose=False, summary=False)
    # print(model.summary())
    # quit()

    ROI_keys = [#'MFG', 'IFG', 'SFG',
                #'ATL',
                # 'ITG', 'MTG', 'FuG', 'PhG', 'pSTS', 'SPL',
                #'IPL', 'Pcun', #'PCC',
                'LOC', 'EVC', 'FuG', #EVC is toxic?
                ]
    ROI_keys2 = ['SFG', 'MFG' 'IFG', ]# 'IPL', 'ATL', 'FuG', 'ITG']
    ROIs_vnd = []
    ROIs_ERS_vnd = []
    ROIs_mvpa_vnd = []
    for ROI in ROIs:
        for ROI_key in ROI_keys:
            if ROI_key in ROI:
                ROIs_vnd.append(ROI)
                break

    for ROI in ERS_obj_ROIs:
        for ROI_key in ROI_keys:
            if ROI_key in ROI:
                ROIs_ERS_vnd.append(ROI)
                break
    for ROI in ROIs_mvpa:
        for ROI_key in ROI_keys2:
            if ROI_key in ROI:
                ROIs_mvpa_vnd.append(ROI)
                break

    print(f'{ROIs_mvpa_vnd=}')
    print(f'{ROIs_ERS_vnd=}')
    df['vnd_RSA'] = df[ROIs_vnd].mean(axis=1)
    # print(df['vnd_RSA'])
    # quit()
    df['vnd_ERS'] = df[ROIs_ERS_vnd].mean(axis=1)
    df['vnd_MVPA'] = df[ROIs_mvpa_vnd].mean(axis=1)

    df_agg = df.groupby('sn').mean()
    df_agg['RSA_ERS'] = df_agg['vnd_RSA'] + df_agg['vnd_ERS']
    df_agg = df_agg[['smlr_ci', 'vnd_RSA', 'vnd_ERS', 'vnd_MVPA',
                     'vendor', 'age']]

    ROIs = ['vnd_RSA'] + ROIs
    ERS_ROIs = ['vnd_ERS'] + ERS_ROIs

    pd.set_option('display.precision', 3)

    # vnd_MVPA + vnd_RSA d_M + v_M + brain_M  +
    #  + all_M + brain_M  + dv_pos + dv_ant
    #  'dv_pos + dv_ant + inc' \

    # dv_pos + dv_ant +
    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    # print(df['age']) # dv_ant + dv_pos + inc + all_M +
    formula_gen = 'hit_hit ~ 1 + inc + dd + {ROI} ' \
                  '+ (1 + inc + dd | sn)'

    # formula_gen = 'dv_pos ~ 1 + dd + all_M + brain_M  + vv + dv_ant + inc ' \
    #               '+ (1  | sn)'
    # formula_gen = 'vnd_MVPA ~ 1 + vendor ' \
    #               ' + (1 | sn)'
    # cols = get_formula_cols(df, formula_gen)
    # print(f'{cols=}')
    # df_vals = df[cols].dropna()
    # print(f'{len(df_vals)=}')
    # model = Lmer(formula_gen, data=df_vals)
    # model.fit(REML=True, verbose=False, summary=False)
    # # summary = model.coefs
    # print(model.summary())
    # quit()


    for ROI in ROIs:
        formula = formula_gen.format(ROI=ROI)

        cols = get_formula_cols(df, formula)

        df_vals = df[cols].dropna()
        for col in cols:
            if col in ['inc', 'sn', 'con_hit', 'hit_hit', 'vis_hit']:
                continue
            df_vals[col] = stats.zscore(df_vals[col])
        model = Lmer(formula, data=df_vals)
        model.fit(REML=True, verbose=False, summary=False)
        summary = model.coefs
        print(summary)
        p = summary['P-val'].loc[ROI]
        t = summary['T-stat'].loc[ROI]
        print(f'{ROI}: {p=:.4f}, {t=:+.3f}')
        # quit()

        if p < .05:
            print(model.summary())
            print('-'*100)
            # print(model.anova())
            # print('-'*100)
            # print('-'*100)
    quit()


if __name__ == '__main__':
    # plot_meta_corr_matrix()
    do_RSA_x_vendor()
