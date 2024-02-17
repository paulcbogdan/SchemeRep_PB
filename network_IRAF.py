import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from atlas_utils import get_atlas
from utils import get_RSA_fn, get_formula_cols

# from connsearch.report.plots import plot_ROI_scores
import scipy.stats as stats




def export_IRAF_csv(age=2, early=True, semantic=True, inc=None,
                 bilateral=False, combine_regions=True,
                 vec_prod=False, PCA_obj=True,
                 org_by_region=False, rxr=False,
                 run_lmer=False,
                 DNN_layer=2, fp_fMRI_col='obj7_fMRI',
                 verbose=True, fp=None, require_all_sns=True,
                 req_all_N=False, key='scn'):
    if fp is None:
        fn = get_RSA_fn(inc=inc, age=age, semantic=semantic,
                        DNN_layer=DNN_layer,
                         fp_fMRI_col=fp_fMRI_col, PCA_obj=PCA_obj,
                         bilateral=bilateral, combine_regions=combine_regions,
                         vec_prod=vec_prod, org_by_region=org_by_region,
                         )
        fp = fr'cache/RSA/{fn}.pkl'

    with open(fp, 'rb') as file:
        d = pickle.load(file)
    atlas = get_atlas(combine_regions=combine_regions or org_by_region,
                      combine_bilateral=bilateral or org_by_region)

    df_as_d = defaultdict(list)
    n_trials = d['IRAFs_ROI'][key]['IPL_L'].shape[-1]
    sns = np.repeat(np.array(d['sns'])[:, None], n_trials, axis=1)
    sns = np.reshape(sns, -1)
    df_as_d['sn'] = sns
    bhv_cols = ['hit_hit', 'vis_hit', 'con_hit', 'inc']
    for col in bhv_cols:
        df_as_d[col].extend(list(np.reshape(d['bhv'][col], -1)))
    df = pd.DataFrame(df_as_d)
    all_regions = []
    for ROI, region in zip(atlas['ROIs'], atlas['ROI_regions']):
        IRAFs = d['IRAFs_ROI'][key][ROI]
        IRAFs = np.reshape(IRAFs, -1)
        df[ROI] = IRAFs
        if f'{region}_R' in df.columns:
            df[region] = df[f'{region}_L'] + df[f'{region}_R']
            all_regions.append(region)
    # df['inc'] = df['inc'].apply(lambda x: 'i' if x == 1 else
    #                                   'n' if x == 2 else 'c')

    all_regions_str = '+ '.join(all_regions)
    df = df[df['inc'] != 2]

    from pymer4.models import Lmer
    formula = f'hit_hit ~ IPL*MFG*LOC*ATL + (1 | sn)'
    # formula = f'inc ~ 1 + {all_regions_str} + (1 | sn)'
    # formula = f'inc ~ 1 + (1 | sn)'

    # df.dropna(subset=['hit_hit'], inplace=True)
    df.dropna(subset=all_regions, inplace=True)

    # formula = f'vis_hit ~ inc + (1 + inc | sn)'
    # cols = get_formula_cols(df, formula)
    # cols = list(set(cols).union({'IPL', 'MFG', 'LOC', 'ATL'}))
    # print(cols)
    # df.dropna(subset=cols, inplace=True)
    x_cols = get_formula_cols(df, formula.split('~')[1])
    for col in x_cols:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].astype('category')
            continue
        n_nans = np.sum(pd.isna(df[col]))
        print(f'{col}: {n_nans=} | {df[col].dtypes}')
        df[col] = stats.zscore(df[col], nan_policy='omit')
    # plt.scatter(df['MFG'], df['IPL'])
    # plt.show()
    model = Lmer(formula, data=df)
    model.fit(REML=False, verbose=False, summary=True)
    summary = model.coefs
    print(summary)


if __name__ == '__main__':
    export_IRAF_csv()