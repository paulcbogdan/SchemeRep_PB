import os

from activity_ERS import get_df_ERS, get_df_trialwise_MVPA
from trialwise_graph_theory import get_df_trial_graphs
from ven_x_dor import get_vendor_df

os.chdir('C:\PycharmProjects_C\SchemeRep')

import numpy as np
import pandas as pd

from atlas_utils import get_atlas

from utils import pickle_wrap, get_RSA_fn, timing, get_formula_cols
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
        df[f'{ROI}_rsa'] = IRAFs
        ROI_cols.append(ROI)
        if f'{region}_R' in df.columns:
            df[region] = df[f'{region}_L_rsa'] + df[f'{region}_R_rsa']
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
        df, RSA_cols = prep_IRAF_df_age(age=age, semantic=semantic, inc=inc,
                                        bilateral=bilateral,
                                        combine_regions=combine_regions,
                                        vec_prod=vec_prod, PCA_obj=PCA_obj,
                                        org_by_region=org_by_region,
                                        DNN_layer=DNN_layer,
                                        fp_fMRI_col=fp_fMRI_col, fp=fp, key=key)
        df['age'] = age
        df_l.append(df)
    df = pd.concat(df_l)
    return df, RSA_cols


def get_super_df(fp='obj7_fMRI', fp_col=False):
    kwargs = {
        'fp_fMRI_col': fp,
        'key': 'scn' if 'scn' in fp else 'obj',
        'semantic': True,
        'combine_regions': True
    }
    df, RSA_cols = pickle_wrap(None, get_trialwise_RSA, kwargs=kwargs,
                               cache_dir='cache', easy_override=False)
    df.set_index(['sn', 'obj'], inplace=True)


    fps_all = ['bl7_fMRI', 'obj7_fMRI', 'scn7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    fp2ERS_ROIs = {}
    for fp1 in fps_all:
        if fp1 == fp: continue
        df_ERS, ERS_ROIs, _ = pickle_wrap(None, get_df_ERS,
                                          kwargs={'fp0': fp, 'fp1': fp1},
                                          cache_dir='cache', easy_override=True)
        df_ERS.set_index(['sn', 'obj'], inplace=True)
        df = df.join(df_ERS[ERS_ROIs])
        fp2ERS_ROIs[fp1] = ERS_ROIs


    df_vdr, vndr_cols = pickle_wrap(None, get_vendor_df, kwargs={'fp': fp},
                                    cache_dir='cache', easy_override=False)
    df_vdr.set_index(['sn', 'obj'], inplace=True)
    if fp_col:
        vdr_key_cols = [f'{col}_{fp}' for col in vndr_cols]
        col2fp_col = {col: key_col for col, key_col
                      in zip(vndr_cols, vdr_key_cols)}
        df_vdr.rename(columns=col2fp_col, inplace=True)
        vndr_cols = vdr_key_cols
    df = df.join(df_vdr[vndr_cols])

    # df_MVPA, ROIs_mvpa, _, _ = get_df_trialwise_MVPA(fp, key='inc', vals=(1, 3),
    #                                                  combine_regions=True)
    # df_MVPA.set_index(['sn', 'obj'], inplace=True)
    # df = df.join(df_MVPA[ROIs_mvpa])
    # df.reset_index(inplace=True, drop=False)
    # ERS_ROIs, ERS_obj_ROIs,
    return df, RSA_cols, fp2ERS_ROIs, vndr_cols


def do_RSA_x_vendor():
    dfs_l = []

    # for fp in ['obj7_fMRI', 'scn7_fMRI', 'bl7_fMRI',
    #            'con7_fMRI', 'vis7_fMRI']:
    #     df, RSA_cols, fp2ERS_ROIs, vndr_cols = get_super_df(fp=fp)
    #     dfs_l.append(df)
    # df = pd.concat(dfs_l)
    df, RSA_cols, fp2ERS_ROIs, vndr_cols = get_super_df(fp='obj7_fMRI')
    df_graph, new_cols = get_df_trial_graphs(fp='obj7_fMRI')

    # print(f'{RSA_cols=}')
    # quit()
    #
    # from pymer4 import Lmer
    #
    # ROI_keys = [  # 'MFG', 'IFG', 'SFG',
    #     # 'ATL',
    #     # 'ITG', 'MTG', 'FuG', 'PhG', 'pSTS', 'SPL',
    #     # 'IPL', 'Pcun', #'PCC',
    #     'LOC', 'EVC',  # 'FuG', #EVC is toxic? 'sOcG',
    # ]
    # ROI_keys2 = ['SFG', 'MFG' 'IFG', ]  # 'IPL', 'ATL', 'FuG', 'ITG']
    # ROIs_vnd = []
    # ROIs_ERS_vnd = []
    # ROIs_mvpa_vnd = []
    # for ROI in ROIs:
    #     for ROI_key in ROI_keys:
    #         if ROI_key in ROI:
    #             ROIs_vnd.append(ROI)
    #             break
    #
    # # for ROI in ERS_obj_ROIs:
    # #     for ROI_key in ROI_keys:
    # #         if ROI_key in ROI:
    # #             ROIs_ERS_vnd.append(ROI)
    # #             break
    # for ROI in ROIs_mvpa:
    #     for ROI_key in ROI_keys2:
    #         if ROI_key in ROI:
    #             ROIs_mvpa_vnd.append(ROI)
    #             break
    #
    # print(f'{ROIs_mvpa_vnd=}')
    # print(f'{ROIs_ERS_vnd=}')
    # df['vnd_RSA'] = df[ROIs_vnd].mean(axis=1)
    # # print(df['vnd_RSA'])
    # # quit()
    # df['vnd_ERS'] = df[ROIs_ERS_vnd].mean(axis=1)
    # df['vnd_MVPA'] = df[ROIs_mvpa_vnd].mean(axis=1)
    #
    # df_agg = df.groupby('sn').mean()
    # df_agg['RSA_ERS'] = df_agg['vnd_RSA'] + df_agg['vnd_ERS']
    # df_agg = df_agg[['smlr_ci', 'vnd_RSA', 'vnd_ERS', 'vnd_MVPA',
    #                  'vendor', 'age']]
    #
    # ROIs = ['vnd_RSA'] + ROIs
    # ERS_ROIs = ['vnd_ERS'] + ERS_ROIs

    # pd.set_option('display.precision', 3)

    # vnd_MVPA + vnd_RSA d_M + v_M + brain_M  +
    #  + all_M + brain_M  + dv_pos + dv_ant
    #  'dv_pos + dv_ant + inc' \

    # dv_pos + dv_ant +
    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    # print(df['age']) # dv_ant + dv_pos + inc + all_M +
    formula_gen = 'vnd_RSA ~ 1 + dd + vv +' \
                  '+ (1 + dd + vv  | sn)'

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
        quit()
        p = summary['P-val'].loc[ROI]
        t = summary['T-stat'].loc[ROI]
        # print(f'{ROI}: {p=:.4f}, {t=:+.3f}')
        # quit()

        if p < .05:
            print(model.summary())
            print('-' * 100)
            # print(model.anova())
            # print('-'*100)
            # print('-'*100)
    quit()


if __name__ == '__main__':
    # plot_meta_corr_matrix()
    do_RSA_x_vendor()
