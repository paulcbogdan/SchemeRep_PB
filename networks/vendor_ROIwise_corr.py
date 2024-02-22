from analyze_rs import get_rs_vendor_df
from atlas_utils import get_atlas
from old.plot_gen import my_plot_surf
from utils import pickle_wrap
from vendor_lmers import get_vendor_df
import scipy.stats as stats
import matplotlib.pyplot as plt

from vendor_partitioning import get_vendor_partitions


def do_vendor_ROIwise(fp='obj7_fMRI', base='dd', seed='dp'):
    # df, vndr_cols = pickle_wrap(get_vendor_df, None,
    #                             kwargs={'fp': fp, 'scrub': False, 'anat': True,
    #                                     'roiwise': True},
    #                             easy_override=False, cache_dir='cache')

    df, vndr_cols = pickle_wrap(get_rs_vendor_df, kwargs={'roiwise': True},
                                easy_override=False)

    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False)

    scores = []
    base2p = {'dd': p_dorsal, 'vv': p_ventral,
              'dv_ant': p_d_ant + p_v_ant, 'dv_pos': p_d_pos + p_v_pos}
    for i in range(246):
        seed_roi = f'{seed}_{i}'
        if i in base2p[base]:
            scores.append(0)
            continue

        df_ = df[[seed_roi, base]].dropna()
        r, p = stats.pearsonr(df_[base], df_[seed_roi])
        scores.append(r*100)

    # plt.hist(scores)
    # plt.show()
    # quit()
    atlas = get_atlas()
    my_plot_surf(scores, atlas, f'{base} x {seed}_i',
                 neg='',
                 thresh=10, vmax=50)




def get_ROIwise_df():
    pass


if __name__ == '__main__':
    do_vendor_ROIwise()












