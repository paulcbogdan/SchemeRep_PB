import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from Study3.EEG_simultaneous_funcs import get_EEG_score_sn
from Study3.analyze_plot_Fig7 import prep_Study3, conv
from Study3.fMRI_simultaneous_funcs import get_fMRI_ar
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from Utils.plotting_funcs import plot_connectivity
from marinate.pkld import pkld


@pkld
def get_EEG_x_conn_mat(sn, sess, just_frontal=True,
                       hz_low=4, hz_high=12, ee=True,
                       combine_regions=False):
    ar_fMRI, key2idxs = (
        pickle_wrap(get_fMRI_ar,
                    kwargs={'sn': sn, 'sess': sess,
                            'combine_regions': combine_regions},
                    easy_override=False, verbose=0, ))
    ar_fMRI = stats.zscore(ar_fMRI, axis=-1)
    if ee:
        print(ar_fMRI.shape)
        ee_fMRI = ar_fMRI[None, :, :] * ar_fMRI[:, None, :]
        ee_fMRI = stats.zscore(ee_fMRI, axis=-1)
        ee_vv_l_fMRI = []
        for roi_i in tqdm(range(ee_fMRI.shape[0])):
            ee_v_fMRI = ee_fMRI[roi_i, :, :]
            ee_vv_fMRI = np.abs(ee_v_fMRI[:, None] - ee_v_fMRI[None, :])
            ee_vv_l_fMRI.append(ee_vv_fMRI)
        abs_mat = np.array(ee_vv_l_fMRI)
    else:
        abs_mat = np.abs(ar_fMRI[None, :, :] - ar_fMRI[:, None, :])
    # abs_mat = np.abs(ar_fMRI[None, :, :]) + np.abs(ar_fMRI[:, None, :])
    # abs_mat = ar_fMRI[None, :, :] + ar_fMRI[:, None, :]

    num_TRs = ar_fMRI.shape[-1]

    if just_frontal:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4', ]
    else:
        picks = ['F1', 'Fz', 'F2', 'F3', 'F4',
                 'FC1', 'FCz', 'FC2', 'FC3', 'FC4',
                 'C1', 'Cz', 'C2', 'C3', 'C4',
                 'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
                 'P1', 'Pz', 'P2', 'P3', 'P4']

    kw = {'sn': sn, 'sess': sess,
          'num_TRs': num_TRs, 'picks': picks,
          'custom_freqs': tuple(np.linspace(0.5, 50, 100))}

    EEG_fluc = pickle_wrap(get_EEG_score_sn, kwargs=kw,
                           easy_override=False, verbose=1)

    abs_mat = abs_mat[..., 6:]

    EEG_fluc = np.nanmean(EEG_fluc, axis=0)  # (freq, TR)
    EEG_fluc = EEG_fluc.T  # (TR, freq)

    EEG_fluc = EEG_fluc[6:, :]  # drop edge artifact
    EEG_fluc = conv(EEG_fluc).T

    abs_mat = stats.zscore(abs_mat, axis=-1)
    EEG_fluc = stats.zscore(EEG_fluc, axis=-1)

    v_delta_theta = np.nanmean(EEG_fluc[hz_low:hz_high, :], axis=0)

    conn = (abs_mat @ v_delta_theta) / abs_mat.shape[-1]
    return conn



def make_overall_EEG_x_conn(just_frontal=False, combine_regions=True):
    sns, sesses, bad_sns = prep_Study3()
    corrs_all = []
    for sn in tqdm(sns):
        mats = []
        for sess in sesses:
            if sn in ['01', '02', '03', '09', '18'] and '02' in sess: continue
            if (sn, sess) in bad_sns: continue
            mat = get_EEG_x_conn_mat(sn, sess, just_frontal=just_frontal,
                                     combine_regions=combine_regions)
            mat_down = np.zeros((mat.shape[0] // 2,
                                 mat.shape[1] // 2,
                                 mat.shape[2] // 2))
            for i in range(mat_down.shape[0]):
                for j in range(mat_down.shape[1]):
                    for k in range(mat_down.shape[2]):
                        mat_down[i, j, k] = np.nanmean(
                            mat[2 * i:2 * i + 2, 2 * j:2 * j + 2, 2 * k:2 * k + 2])
            mat = mat_down
            # plt.imshow(mat[20])
            # plt.show()
            # quit()
            # print(mat)
            # quit()
            # M = np.nanmean(mat, axis=0)
            # M_sq = (M[None, :] + M[:, None]) / 2
            # mat -= M_sq
            mats.append(mat)
        if len(mats) == 0: continue
        corr = np.nanmean(mats, axis=0)
        corrs_all.append(corr)
    corrs_all = np.array(corrs_all)
    M = np.nanmean(corrs_all, axis=0)
    SE = np.nanstd(corrs_all, axis=0, ddof=1) / np.sqrt(len(corrs_all))
    corr = M / SE
    atlas = get_atlas(lifu_labels=combine_regions,
                      combine_regions=combine_regions,
                      combine_bilateral=True)
    ROIs = atlas['ROIs']
    if len(corr.shape) == 3:
        for ROI, corr_ROI in zip(ROIs, corr):
            tile = np.nanquantile(corr_ROI, .95)
            corr_ROI[corr_ROI < tile] = np.nan
            plot_connectivity(corr_ROI, atlas['ticks'], atlas['tick_labels'],
                              atlas['tick_lows'], no_avg=True,
                              title=ROI, vmin=-6, vmax=6)
    else:
        plot_connectivity(corr, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], no_avg=True, )


if __name__ == '__main__':
    make_overall_EEG_x_conn()
