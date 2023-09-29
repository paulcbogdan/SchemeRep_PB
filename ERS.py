from tqdm import tqdm

from PFC_analysis import load_and_get_ROI_vecs
from ROIs import get_BN_and_resample, get_atlas
from organize_bhv import get_all_sns, get_trial_info
from single_trial_conn import corr_last_dim
import numpy as np
import scipy.stats as stats

from utils import stdize, pb_outer
import matplotlib.pyplot as plt
from time import time

def ERS_sn(sn, atlas):
    df_sn = get_trial_info(sn)
    keeps = df_sn['vis_type'] == 'old'

    ROI2vecs_enc, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='vis_fMRI')
    ROI2vecs_ret, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='obj_fMRI')
    ers_l = []
    hits = df_sn['hit_bool'].fillna(False).values
    for ROI in atlas['ROIs']:#, desc=f'ERS {sn} by ROI'):
        # if 'LOC' not in ROI and 'EVC' not in ROI:
        #     continue
        try:
            vecs_enc = ROI2vecs_enc[ROI]
            vecs_ret = ROI2vecs_ret[ROI]
        except KeyError:
            ers_l.append(np.nan)
            continue
        if vecs_enc.shape != vecs_ret.shape:
            ers_l.append(np.nan)
            continue
        vecs_enc = vecs_enc[keeps, :]
        vecs_ret = vecs_ret[keeps, :]
        idxs = np.arange(vecs_enc.shape[1])
        np.random.shuffle(idxs)
        n_idx = 100
        vecs_enc = vecs_enc[:, idxs[:n_idx]]
        vecs_ret = vecs_ret[:, idxs[:n_idx]]

        vecs_enc = stdize(vecs_enc, axis=1)
        vecs_ret = stdize(vecs_ret, axis=1)
        vecs_enc = pb_outer(vecs_enc, vecs_enc, tril=True, nan_diag=True)
        vecs_ret = pb_outer(vecs_ret, vecs_ret, tril=True, nan_diag=True)

        # subtract the mean of ers with other trials
        ers = corr_last_dim(vecs_enc, vecs_ret)
        vecs_enc_ = stdize(vecs_enc, axis=1)
        vecs_ret_ = stdize(vecs_ret, axis=1)
        vecs_enc_ = vecs_enc_[:, None, :]
        vecs_ret_ = vecs_ret_[None, :, :]
        vecs_prod = vecs_enc_ * vecs_ret_
        diag = np.diag_indices_from(vecs_prod[:, :, 0])
        vecs_prod[diag[0], diag[1], :] = 0
        corrmat = np.mean(vecs_prod, axis=2)
        # ers_else is a matrix, where i, j is enc[i] x ret[j]
        ers_else = np.sum(corrmat, axis=1) / (corrmat.shape[1] - 1)
        ers = ers - ers_else
        ers_l.append(np.nanmean(ers))
    ers_sn = np.array(ers_l)
    return ers_sn


def ERS_all_sn():
    atlas = get_atlas(combine_regions=True, bilateral=False)
    # print(atlas['ROIs'])
    # quit()
    age2sn = get_all_sns(ret=True)
    ers_l_all = []
    for i, sn in tqdm(enumerate(age2sn[1]), desc='ERS, looping subjects'):
        ers_l_all.append(ERS_sn(sn, atlas))
    ers_all = np.array(ers_l_all)
    print(ers_all.shape)
    for i, ROI in enumerate(atlas['ROIs']):
        ers = ers_all[:, i]
        M = np.nanmean(ers)
        SD = np.nanstd(ers)
        N = len(ers[~np.isnan(ers)])
        SE = SD / np.sqrt(N)
        t = M / SE
        p = stats.t.sf(np.abs(t), len(ers)-1)
        # if t > 2:
            # print(ers)
        print(f'{ROI}, t[{N-1}]={t:.2f}, p={p:.3f}')


if __name__ == '__main__':
    ERS_all_sn()