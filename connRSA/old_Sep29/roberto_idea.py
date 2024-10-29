from datetime import datetime

from tqdm import tqdm

from atlas_utils import get_atlas

from connRSA.conn_Fig6 import get_cross_ERS_mat, get_idxs, get_cross_IRAF_mat
from connRSA.single_trial_conn import prep_fps
from utils import pickle_wrap
import numpy as np
from scipy import stats

import os
os.chdir(r'H:\PycharmProjects_H\SchemeRep')
import matplotlib.pyplot as plt

def plot_NPS_spectrum():
    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)
    ERS_nan_block = False

    cross = False

    drop_con = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    atlas = get_atlas()

    idxs = get_idxs('IT')
    coords = atlas['coords']
    regions = atlas['ROI_regions']
    IRAF = True

    # T_idxs_sorted = sorted(idxs, key=lambda x: coords[x][1])
    # print(idxs_sorted)
    # for idx in T_idxs_sorted:
    #     print(f'{idx}: {regions[idx]} | {coords[idx][1]}')

    # EVC_idxs = get_idxs('EVC')
    # LOC_idxs = get_idxs('LOC')
    # sOcG_idxs = get_idxs('sOcG')
    # late_OC_idxs_sorted = sorted(LOC_idxs + sOcG_idxs,
    #                              key=lambda x: coords[x][1])
    # idxs_sorted = EVC_idxs + late_OC_idxs_sorted + T_idxs_sorted

    idxs = get_idxs('ITL')
    idxs_sorted = sorted(idxs, key=lambda x: coords[x][1])

    ROIs = atlas['ROIs']

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']

    four_tasks = '7'
    fps = prep_fps(four_tasks)
    fps = [fp for fp in fps if 'con' not in fp]

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]

    ERS_scores_all = []
    for i, sn in tqdm(enumerate(sns), desc=f'Looping ERS_scores: {cross=}'):
        # print(f'Onto: {sn}')
        kwargs['sn'] = sn
        kwargs['fps'] = fps

        if IRAF:
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = True
            kwargs['just_get_IRAFs'] = True
            ERS_scores = pickle_wrap(get_cross_IRAF_mat, kwargs=kwargs, verbose=-1,
                                     easy_override=False, dt_max=dt_max)
        else:
            kwargs['cross'] = cross
            kwargs['nan_block'] = ERS_nan_block
            kwargs['get_var'] = 'all'
            ERS_scores = pickle_wrap(get_cross_ERS_mat, kwargs=kwargs, verbose=-1,
                                     easy_override=False, dt_max=dt_max)
        ERS_scores = np.array(ERS_scores)
        ERS_scores = np.nanmean(ERS_scores, axis=0) # average over fps
        ERS_scores = ERS_scores[idxs_sorted]
        ERS_scores_all.append(ERS_scores)


    ERS_scores_all = np.array(ERS_scores_all)
    # print(ERS_scores_all.shape)
    # quit()
    ERS_scores_t = ((np.nanmean(ERS_scores_all, axis=0) /
                    np.nanstd(ERS_scores_all, axis=0)) *
                    np.sqrt(ERS_scores_all.shape[0]))
    # print(ERS_scores_t)
    # quit()
    # ERS_scores = np.nanmean(ERS_scores_all, axis=0)
    ERS_scores = ERS_scores_t
    ERS_scores = (ERS_scores > 2).astype(int)
    ERS_scores = np.cumsum(ERS_scores, axis=0)
    ERS_scores[ERS_scores > 10] = 10

    vmin = np.nanquantile(ERS_scores, 0.05)
    vmax = np.nanquantile(ERS_scores, 0.95)
    vmin = -vmax
    plt.imshow(ERS_scores, aspect='auto', interpolation='none',
               vmin=vmin, vmax=vmax, cmap='viridis', )
    cbar = plt.colorbar()
    cbar.set_label('t-value', rotation=270, labelpad=12)
    plt.ylabel('ROIs')
    idxs_ticks = np.arange(len(idxs_sorted))[::5]
    region_ticks = [f'{regions[idx]}' for idx in idxs_sorted[::5]]
    plt.yticks(idxs_ticks, region_ticks)
    plt.xlabel('Item')
    plt.show()

    M_p = []
    for i, ROI in enumerate(idxs_sorted):
        wilk, p = stats.shapiro(ERS_scores[i], nan_policy='omit')
        t, _ = stats.ttest_1samp(ERS_scores[i], 0)
        print(f'{ROI}, {regions[ROI]}: wilk p = {p:.3f} | t = {t=:.2f}')
        M_p.append(p)
    M_p = np.array(M_p)
    print(f'Overall: {M_p.mean():.3f}')

    # for j in range(ERS_scores.shape[1]):
    #     wilk, p = stats.shapiro(ERS_scores[:, j], nan_policy='omit')
    #     print(f'Item {j}: wilk p = {p:.3f}')

if __name__ == '__main__':
    plot_NPS_spectrum()

