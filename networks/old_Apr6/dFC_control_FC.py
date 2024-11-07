from analyze_rs import get_rs_vendor_df

import pandas as pd

from vendor_lmers import get_vendor_df
from utils import get_formula_cols
from Utils.pickle_wrap_funcs import pickle_wrap
import scipy.stats as stats


# os.chdir(r'/')

def get_all_task_vendor(zscore=True, anat=False, roiwise=True):
    fps = ['bl7_fMRI', 'con7_fMRI', 'rs', 'vis7_fMRI', 'obj7_fMRI']
    df_l = []
    for fp in fps:
        if fp == 'rs':
            if anat:
                # df, networks = pickle_wrap(get_df_networks,
                #                            kwargs={'fp': 'rs',
                #                                    'norm_std': False,
                #                                    'zscore': zscore},
                #                            easy_override=False)
                df, vndr_cols = pickle_wrap(get_rs_vendor_df,
                                            kwargs={'roiwise': roiwise,
                                                    'do_hemi': False,
                                                    'zscore': True,
                                            'high_var_confounds': False},
                                            easy_override=False)
                df['task'] = 'RS'
        else:
            df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                        kwargs={'fp': fp, 'scrub': False,
                                                'roiwise': roiwise,
                                                'zscore': zscore, 'anat': anat},
                                        easy_override=False)
            df['task'] = fp.split('_')[0][:-1].upper()
        df_l.append(df)
    df = pd.concat(df_l).reset_index(drop=True)
    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']


    # df['dd_vv'] = stats.zscore(df['dd_vv'])
    # df['dv_dv'] = stats.zscore(df['dv_dv'])
    # df['vv'] = stats.zscore(df['vv'])
    return df


def vendor_lmer_BL_resp(fp='bl7_fMRI'):


    df = get_all_task_vendor(zscore=True, roiwise=False)


    df['dp_va'] = df['dp'] * df['va']
    df['dp_va'] = stats.zscore(df['dp_va'], nan_policy='omit')
    df['da_vp'] = df['da'] * df['vp']
    df['da_vp'] = stats.zscore(df['da_vp'], nan_policy='omit')

    df['vv'] = stats.zscore(df['vv'], nan_policy='omit')
    df['dd'] = stats.zscore(df['dd'], nan_policy='omit')

    formula = ('dv_pos ~ vv * task + '
               '(1 | sn)')

    # However, this is only the case for RS
    # formula = ('dv_pos ~ vv * dp_va + '
    #            '(1 + vv * dp_va | sn)')

    formula = ('dv_pos ~ dd * da_vp + '
               '(1 + dd * da_vp | sn)') # sum of dd + da_vp = dd:da_vp coef

    # formula = ('dp ~ va * task + '
    #            '(1 | sn)')

    # formula = ('dv_pos ~ dd + FC_all + '
    #            '(1 + dd + FC_all | sn)')

    # formula = ('dd_vv ~ dv_dv * task +'
    #            '(1 | sn)')

    from pymer4 import Lmer
    cols = get_formula_cols(df, formula)
    df_vals = df[cols].dropna()
    for col in cols:
        if 'sn' in col or 'task' in col: continue
        df_vals[col] = stats.zscore(df_vals[col])
    model = Lmer(formula, data=df_vals)
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())


if __name__ == '__main__':
    vendor_lmer_BL_resp()