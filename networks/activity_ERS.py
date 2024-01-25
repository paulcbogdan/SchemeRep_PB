import pandas as pd
from pymer4 import Lmer
from tqdm import tqdm

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from network_funcs import load_FC_for_Lifu
from org_sns import get_all_sns
from organize_bhv import get_trial_info
from univariate_activity import keys_from_formula
from utils import pickle_wrap
import numpy as np

from functools import wraps
from time import time
import matplotlib.pyplot as plt
import scipy.stats as stats

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

@timing
def do_activity_ERS_sn(sn, fp0='bl3_fMRI', fp1='obj7_fMRI',
                       combine_regions=False):
    atlas = get_atlas(combine_regions=combine_regions)
    df_sn = get_trial_info(sn)
    kwargs0 = {
        'sn': sn,
        'atlas': atlas,
        'fp_fMRI_col': fp0,
        'df_sn': df_sn,
        'nan_thresh': 1.01,
        'drop_nan_voxels': False,
        'org_by_region': False,
        'easy_override': False,
        'combine_regions': combine_regions,
    }
    ROI2vecs0 = get_ROI_vecs(**kwargs0)
    kwargs1 = kwargs0 | {'fp_fMRI_col': fp1}
    ROI2vecs1 = get_ROI_vecs(**kwargs1)
    if combine_regions:
        assert 'SFG_L' in ROI2vecs0, f'{ROI2vecs0.keys()=}'
        assert 'SFG_L' in ROI2vecs1, f'{ROI2vecs1.keys()=}'
    ERSs = []
    cols = []
    for ROI, vecs0 in ROI2vecs0.items():
        vecs1 = ROI2vecs1[ROI]
        corr = vecs0[None, :, :] * vecs1[:, None, :]
        corr = np.nanmean(corr, axis=-1)
        same = np.diag(corr.copy()) # copy needed to avoid next line from NaNing
        corr[np.diag_indices_from(corr)] = np.nan
        dif = np.nanmean(corr, axis=0)
        ERS = same - dif
        ERSs.append(ERS)
        ROI_clean = ROI.replace(' ', '_')
        cols.append(f'ERS_{fp0[:2]}_{fp1[:2]}_{ROI_clean}')
    # setting df_sn[cols]=ERS.T instead gives a fragmentation warning
    df_cols = pd.DataFrame(np.array(ERSs).T, columns=cols, index=df_sn.index)
    df_sn = pd.concat([df_sn, df_cols], axis=1)
    df_sn['sn'] = sn
    return df_sn, cols

def make_ERS_df(fp0='bl3_fMRI', fp1='obj7_fMRI', combine_regions=True):
    age2sn = get_all_sns('all', sh=False)
    sns = age2sn[2]
    dfs = []
    ERS_cols = []
    for sn in tqdm(sns, desc='Making ERS dfs'):
        cmb_str = '_cmb' if combine_regions else ''
        fp_pkl = f'cache/ERS_df_{sn}_{fp0}_{fp1}{cmb_str}.pkl'
        df_sn, ERS_cols = pickle_wrap(fp_pkl, lambda: do_activity_ERS_sn(sn,
                combine_regions=combine_regions),
                                      easy_override=True)
        df_sn['sn'] = sn
        dfs.append(df_sn)
        # if len(dfs) > 5:
        #     break
    df = pd.concat(dfs, axis=0)
    for col in ERS_cols:
        df_grp = df.groupby(['sn', 'inc'])[col].mean()
        t, p = stats.ttest_rel(df_grp.loc[:, 1], df_grp.loc[:, 3],
                               nan_policy='omit')
        df_grp = df.groupby(['sn'])[col].mean()
        t0, p0 = stats.ttest_1samp(df_grp, 0, nan_policy='omit')
        if (p < .01) | (p0 < .01):
            print(f'{col=}')
            print(f'\t Inc effect: {t=:.2f}, {p=:.3f}')
            print(f'\t Above zero: {t0=:.2f}, {p0=:.3f}')
        continue
        formula = f'{col} ~ 1 + inc + (1 | sn)'
        keys = keys_from_formula(formula)
        model = Lmer(formula, data=df[keys].dropna())
        model.fit(REML=True, verbose=False, summary=False)
        result = model.summary()

        if (result['P-val'].iloc[0] < .01) | (result['P-val'].iloc[1] < .01):
            print(f'{col=}')
            print(result)
            print('-'*100)

if __name__ == '__main__':
    make_ERS_df()






