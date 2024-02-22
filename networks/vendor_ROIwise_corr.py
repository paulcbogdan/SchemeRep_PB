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

def do_vendor_ROIwise(fp='rs', base='vv', exclude='va', seed='va'):

    if fp == 'rs':
        df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True,
                                                              'do_hemi': False},
                                    easy_override=False)
    else:
        df, vndr_cols = pickle_wrap(get_vendor_df, None,
                                    kwargs={'fp': fp, 'scrub': False,
                                            'anat': True,
                                            'roiwise': True},
                                    easy_override=False, cache_dir='cache')
    # print(df['ad_no'])
    # quit()

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False)

    df['dd_vv'] = df['dd'] + df['vv']

    scores = []
    base2p = {'dd': p_dorsal, 'vv': p_ventral,
              'dv_ant': p_d_ant + p_v_ant, 'dv_pos': p_d_pos + p_v_pos}

    # pd_no + ad_no + pv_no + av_no +
    formula_gen = ('{base} ~ '
                   '1 + {seed_roi} + FC_all')
    for i in range(246):
        seed_roi = f'{seed}_{i}'
        # if i in base2p[exclude]:
        #     scores.append(0)
        #     continue

        formula = formula_gen.format(base=base, seed_roi=seed_roi)
        cols = get_formula_cols(df, formula)
        df_ = df[cols].dropna()

        model = smf.ols(formula=formula, data=df_)
        res = model.fit()
        t = res.tvalues.loc[f'{seed_roi}']
        scores.append(t)
        # print(res.summary())
        # quit()
        # r, p = stats.pearsonr(df_[base], df_[seed_roi])
        # scores.append(r*100)

    # score = np.array(scores)
    # if np.max(np.abs(scores)) > 10:
    #     factor = np.max(np.abs(scores)) / 10
    #     scores /= factor

    # plt.hist(scores)
    # plt.show()
    # quit()
    atlas = get_atlas()
    my_plot_surf(scores, atlas, f'{fp}: {base} x {seed}_i',
                 neg='',
                 thresh=2, vmax=10)




def get_ROIwise_df():
    pass


if __name__ == '__main__':
    do_vendor_ROIwise()












