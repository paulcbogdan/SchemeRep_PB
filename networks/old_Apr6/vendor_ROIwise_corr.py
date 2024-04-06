from tqdm import tqdm

from analyze_rs import get_rs_vendor_df
from atlas_utils import get_atlas
from old.plot_gen import my_plot_surf
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
import scipy.stats as stats
import matplotlib.pyplot as plt

from vendor_partitioning import get_vendor_partitions
import statsmodels.formula.api as smf
from utils import get_formula_cols
import numpy as np

# def do_vendor_ROIwise(fp='rs', base='vv', exclude='va', seed='va'):
# def do_vendor_ROIwise(fp='rs', base='vv', exclude='va', seed='vp'):
# def do_vendor_ROIwise(fp='rs', base='vv', exclude='va', seed='va'):
def do_vendor_ROIwise(fp='obj7_fMRI', base='dv_pos', exclude='va', seed='dp'):

    if fp == 'rs':
        df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True,
                                                              'do_hemi': False,
                                    'high_var_confounds': False},
                                    easy_override=False)
    else:
        df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                    kwargs={'fp': fp, 'scrub': False,
                                            'anat': True,
                                            'roiwise': True},
                                    easy_override=True, cache_dir='cache')

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False)

    df['va_else'] = df['av_else']
    df['vp_else'] = df['pv_else']
    df['da_else'] = df['ad_else']
    df['dp_else'] = df['pd_else']
    # df['vp_no'] = df['pv_no']
    # df['va_no'] = df['av_no']
    # df['dp_no'] = df['pd_no']

    df['dd_vv'] = df['dd'] + df['vv']

    scores = []
    base2p = {'dd': p_dorsal, 'vv': p_ventral,
              'dv_ant': p_d_ant + p_v_ant, 'dv_pos': p_d_pos + p_v_pos}

    # pd_no + ad_no + pv_no + av_no +

    seed_cols = [f'{seed}_{i}' for i in range(246)]
    df['seed_all'] = df[seed_cols].mean(axis=1)

    noise_cols = [f'{seed}_{i}' for i in range(220, 245)]
    df[f'{seed}_noise'] = df[noise_cols].mean(axis=1)

    # when I regress {seed}_else that essentially makes everything seed relative
    #   to the effect with everything else
    formula_gen = ('{base} ~ 1 + {seed_roi} + FC_all') #  + {seed}_else

    # for i in range(246):
    #     seed_roi = f'{seed}_{i}'
    #     # if i in base2p[exclude]:
    #     #     scores.append(0)
    #     #     continue
    #
    #     formula = formula_gen.format(base=base, seed_roi=seed_roi, seed=seed)
    #     cols = get_formula_cols(df, formula)
    #     df_ = df[cols].dropna()
    #
    #     model = smf.ols(formula=formula, data=df_)
    #     res = model.fit()
    #     t = res.tvalues.loc[f'{seed_roi}']
    #     scores.append(t)
    #
    #
    # score = np.array(scores)
    # print(f'Min score: {np.min(score)} | max: {np.max(score)}')
    # if np.min(np.abs(scores)) > 10:
    #     factor = np.max(np.abs(scores)) / 10
    #     scores /= factor
    #     print(f'Factor divide: {factor}')

    # atlas = get_atlas()
    # my_plot_surf(scores, atlas, f'{fp}: {formula}',
    #              neg='',
    #              thresh=2, vmax=50)
    print(list(df.columns))
    scores = []
    formula_ctrl = ('va ~ 1 + {seed_roi}*da*dp') #  + {seed}_else  + FC_all
    for i in tqdm(range(246), desc='Fitting roiwise ols'):
        seed_roi_M = f'M_{i}'
        # print(df[seed_roi_M])
        formula = formula_ctrl.format(seed_roi=seed_roi_M)
        cols = get_formula_cols(df, formula)
        df_ = df[cols].dropna()
        # print(df_)
        model = smf.ols(formula=formula, data=df_)
        res = model.fit()
        # print(res.summary())
        # quit()
        t = res.tvalues.loc[f'{seed_roi_M}:da:dp']

        # t = res.tvalues.loc[f'{seed_roi_M}']
        scores.append(t)
    atlas = get_atlas()
    my_plot_surf(scores, atlas, f'{fp}: {formula_ctrl}',
                 neg='',
                 thresh=5, vmax=10)


def get_ROIwise_df():
    pass


if __name__ == '__main__':
    do_vendor_ROIwise()












