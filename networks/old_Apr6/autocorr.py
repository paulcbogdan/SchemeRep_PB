import pandas as pd
import scipy.stats as stats

from utils import pickle_wrap
from vendor_lmers import get_vendor_df

def add_prev(df, key, sess):
    for sn, df_sn in df.groupby('sn'):
        if f'{sess.lower()}_trial' in df_sn.columns:
            df_sn = df_sn.sort_values(f'{sess}_trial')
        elif sess.lower() != 'rs':
            raise ValueError
        # df_sn[key] = stats.zscore(df_sn[key])
        df_sn[f'{key}_prev'] = df_sn[key].shift(1)
        df.loc[df_sn.index, f'{key}_prev'] = df_sn[f'{key}_prev']

def add_next(df, key, sess):
    for sn, df_sn in df.groupby('sn'):
        if f'{sess}_trial' in df_sn.columns:
            df_sn = df_sn.sort_values(f'{sess}_trial')
        # df_sn[key] = stats.zscore(df_sn[key])
        df_sn[f'{key}_next'] = df_sn[key].shift(-1)
        df.loc[df_sn.index, f'{key}_next'] = df_sn[f'{key}_next']

def test_vendor_corr(fp='obj7_fMRI', plot=True, hemi=True, scrub=False,
                     anat=True):
    df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                kwargs={'fp': fp, 'scrub': scrub, 'anat': anat,
                                        'hemis': hemi}, easy_override=False,
                                cache_dir='cache')

    names = ['dd', 'vv', 'dv_ant', 'dv_pos',
             'pd_M', 'ad_M', 'pv_M', 'av_M']
    for name in names:
        add_prev(df, name, 'obj')
    # add_prev(df, 'dd', 'obj')
    # print(df[['dd_prev', 'dd', 'obj_trial']])
    # quit()

    formula = ('pd_M ~ dd + dv_ant + dv_pos + '
               'pd_M_prev + ad_M + pv_M + av_M + '
               '(1 | sn)')

    from pymer4 import Lmer
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())



if __name__ == '__main__':
    test_vendor_corr()


