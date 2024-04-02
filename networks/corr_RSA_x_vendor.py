import os

from old.activity_ERS import get_df_ERS
from org_sns import get_sns
from organize_bhv import get_trial_info
from old.trialwise_graph_theory import get_df_trial_graphs
from vendor_lmers import get_vendor_df

os.chdir('E:\PycharmProjects_E\SchemeRep')

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
    cols_pre = df.columns
    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        IRAFs = d['IRAFs_ROI'][key][ROI]
        IRAFs = np.reshape(IRAFs, -1)
        ROI = ROI.replace(' ', '_')
        if not combine_regions:
            ROI = '_'.join(ROI.split('_')[1:])
        df[f'{ROI}_rsa'] = IRAFs
        ROI_cols.append(ROI)
        if f'{region}_R' in df.columns:
            df[f'{region}_rsa'] = df[f'{region}_L_rsa'] + df[f'{region}_R_rsa']
            all_regions.append(region)
    cols_post = df.columns
    new_cols = list(set(cols_post) - set(cols_pre))
    return df, new_cols


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

def get_plain_df_sn(bad_sns=None):
    age2sns = pickle_wrap(get_sns, None, kwargs={'fp_fMRI': 'loose'},
                          easy_override=True)
    df_sns_l = []
    sns = []
    for age, sns_age in age2sns.items():
        if age not in [1, 2]: continue
        sns.extend(sns_age)
    if bad_sns is not None:
        sns = [sn for sn in sns if sn not in bad_sns]
    df_sns_l.extend([get_trial_info(sn, verbose=-1) for sn in sns])
    df_sns = pd.concat(df_sns_l)
    df_sns['age'] = df_sns['sn'].apply(lambda sn: int(str(sn)[0]))
    return df_sns, df_sns_l

def get_super_df(fp='obj7_fMRI', fp_col=False):
    df, _ = get_plain_df_sn()
    df.set_index(['sn', 'obj'], inplace=True)

    kwargs = {
        'fp_fMRI_col': fp,
        'key': 'scn' if 'scn' in fp else 'obj',
        'semantic': True,
        'combine_regions': True
    }
    df_RSA, RSA_cols = pickle_wrap(get_trialwise_RSA, None, kwargs=kwargs, easy_override=True, cache_dir='cache')
    print(df_RSA.columns)
    # print(RSA_cols)
    # quit()
    df_RSA.set_index(['sn', 'obj'], inplace=True)
    df = df.join(df_RSA[RSA_cols])

    fps_all = ['bl7_fMRI', 'obj7_fMRI', 'scn7_fMRI', 'con7_fMRI', 'vis7_fMRI']
    fp2ERS_ROIs = {}
    for fp1 in fps_all:
        if fp1 == fp: continue
        df_ERS, ERS_ROIs, _ = pickle_wrap(get_df_ERS, None, kwargs={'fp0': fp, 'fp1': fp1}, easy_override=True,
                                          cache_dir='cache')
        df_ERS.set_index(['sn', 'obj'], inplace=True)
        df = df.join(df_ERS[ERS_ROIs])
        fp2ERS_ROIs[fp1] = ERS_ROIs


    df_vdr, vndr_cols = pickle_wrap(get_vendor_df, None, kwargs={'fp': fp}, easy_override=False, cache_dir='cache')
    df_vdr.set_index(['sn', 'obj'], inplace=True)
    if fp_col:
        vdr_key_cols = [f'{col}_{fp}' for col in vndr_cols]
        col2fp_col = {col: key_col for col, key_col
                      in zip(vndr_cols, vdr_key_cols)}
        df_vdr.rename(columns=col2fp_col, inplace=True)
        vndr_cols = vdr_key_cols
    df = df.join(df_vdr[vndr_cols])

    return df, RSA_cols, fp2ERS_ROIs, vndr_cols


def do_RSA_x_vendor(fp='obj7_fMRI'):

    df_graph, new_cols = pickle_wrap(get_df_trial_graphs, None, kwargs={'fp': fp}, easy_override=False)

    df, RSA_cols, fp2ERS_ROIs, vndr_cols = pickle_wrap(get_super_df, None, kwargs={'fp': fp}, easy_override=False)

    df_graph.set_index(['sn', 'obj'], inplace=True)
    df = df.join(df_graph[new_cols])
    df.reset_index(inplace=True, drop=False)

    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')
    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    df['dd_clustering'] = stats.zscore(df['dd_clustering'], nan_policy='omit')
    df['vv_clustering'] = stats.zscore(df['vv_clustering'], nan_policy='omit')


    fp_out = fr'csv_out/{fp}_super_first.csv'
    with open(fp_out, 'w') as f:
        df.to_csv(f, index=False)
        quit()

    # print(df['vv_shortest'])
    # print(df['age']) # dv_ant + dv_pos + inc + all_M +
    formula = 'vv_shortest ~ 1 + dd + DP_hemi + DA_hemi +' \
              ' vv + dv_ant + dv_pos + inc' \
              '+ pd_M + ad_M + pv_M + av_M' \
              '+ (1  + dd + DP_hemi + DA_hemi | sn)'
    cols = get_formula_cols(df, formula)
    print(df[cols].head(1))

    from pymer4.models import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())
    quit()



if __name__ == '__main__':
    # plot_meta_corr_matrix()
    do_RSA_x_vendor()
