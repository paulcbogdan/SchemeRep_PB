from Study1A.load_Study1A_funcs import load_FC
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
import numpy as np

from old.plot_gen import my_plot_surf


def plot_inc_ef(combine_regions=True):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True
              }

    if combine_regions:
        kwargs['combine_regions'] = True

    sn_inc_conn, sn_conn, age2idxs, sn_flat, _ = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1,
                    cache_dir='cache')

    print(F'{sn_flat.shape=}')

    sn_flat = np.nanmean(sn_flat, axis=-1)
    sn_flat -= np.nanmean(sn_flat, axis=1)[:, None, :]
    n_sn = sn_flat.shape[0]



    sn_flat = np.reshape(sn_flat, (n_sn * 3, sn_flat.shape[-1]))

    regressors = np.array([[-1, 0, 1] * n_sn]).T
    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    betas = np.dot(XTX_invX, sn_flat)

    Y_pred = np.dot(regressors, betas)
    residual = sn_flat - Y_pred
    sigma_s = np.sum(residual ** 2, axis=0) / (n_sn * 2 - 2)
    ss_x = np.sum(regressors ** 2, axis=0)
    var_beta = sigma_s / ss_x
    z = betas / np.sqrt(var_beta)
    z = z[0, :]
    z *= -1
    print(z)
    # now: positive = incongruent =  red | occipital, ATL, precuneus, maybe MFG
    #      negative = congruent = blue | lateral temporal and lateral parietal

    fp_out = f'result_pics/SuppMat/FigS2A_go.png'
    atlas = get_atlas(combine_regions=combine_regions)
    my_plot_surf(z, atlas, 'Study 1A: Regional PE (BOLD) effects',
                 thresh=2.5, vmax=4, fp_out=fp_out)


if __name__ == '__main__':
    plot_inc_ef()