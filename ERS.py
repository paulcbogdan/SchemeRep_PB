from tqdm import tqdm

from atlas_utils import get_BN_and_resample, get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_all_sns, get_trial_info
from single_trial_conn import corr_last_dim
import numpy as np
import scipy.stats as stats

from utils import stdize, pb_outer
import matplotlib.pyplot as plt
from time import time

def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI'):
    df_sn = get_trial_info(sn)

    ROI2vecs_enc = get_ROI_vecs(sn, atlas, fp0, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False, org_by_region=True)
    ROI2vecs_ret = get_ROI_vecs(sn, atlas, fp1, df_sn, nan_thresh=1.01,
                                drop_nan_voxels=False, org_by_region=True)
    ers_l = []
    # hits = df_sn['hit_bool'].fillna(False).values
    for ROI in atlas['ROI_regions']:#, desc=f'ERS {sn} by ROI'):
        # if 'LOC' not in ROI and 'EVC' not in ROI:
        #     continue
        # print(ROI)
        # quit()
        try:
            vecs_enc = ROI2vecs_enc[ROI]
            vecs_ret = ROI2vecs_ret[ROI]
        except KeyError:
            ers_l.append(np.nan)
            continue
        if vecs_enc.shape != vecs_ret.shape:
            ers_l.append(np.nan)
            print(f'ROI {ROI} has different shapes for enc ({vecs_enc.shape}) '
                  f'and ret ({vecs_ret.shape})')
            continue
        # print(vecs_enc.shape)
        keeps = np.logical_and(~np.isnan(vecs_enc).any(axis=0),
                               ~np.isnan(vecs_ret).any(axis=0))

        vecs_enc = vecs_enc[:, keeps]
        # print(vecs_enc)
        # quit()
        # vecs_enc = np.mean(vecs_enc, axis=1)[:, None]
        vecs_ret = vecs_ret[:, keeps]
        # vecs_ret = np.mean(vecs_ret, axis=1)[:, None]

        # # TODO: divide in random groups of 100 voxels as grid

        # idxs = np.arange(vecs_enc.shape[1])
        # np.random.shuffle(idxs)
        # n_idx = 100
        # vecs_enc = vecs_enc[:, idxs[:n_idx]]
        # vecs_ret = vecs_ret[:, idxs[:n_idx]]
        #
        # vecs_enc = stdize(vecs_enc, axis=1)
        # vecs_ret = stdize(vecs_ret, axis=1)
        vecs_enc = pb_outer(vecs_enc, vecs_enc, tril=True, nan_diag=True)
        vecs_ret = pb_outer(vecs_ret, vecs_ret, tril=True, nan_diag=True)

        # TODO: Adjust to work with euc distance

        # subtract the mean of ers with other trials
        ers = corr_last_dim(vecs_enc, vecs_ret, euc_dist=True)
        vecs_enc_ = stdize(vecs_enc, axis=1)
        vecs_ret_ = stdize(vecs_ret, axis=1)
        vecs_enc_ = vecs_enc_[:, None, :]
        vecs_ret_ = vecs_ret_[None, :, :]
        # vecs_prod = vecs_enc_ * vecs_ret_
        vecs_prod = abs(vecs_enc_ - vecs_ret_)
        diag = np.diag_indices_from(vecs_prod[:, :, 0])
        vecs_prod[diag[0], diag[1], :] = 0
        corrmat = np.mean(vecs_prod, axis=2)
        # ers_else is a matrix, where i, j is enc[i] x ret[j]
        ers_else = np.sum(corrmat, axis=1) / (corrmat.shape[1] - 1)
        ers_dif = ers - ers_else
        # print('test', ers_dif.shape)
        ers_l.append(np.nanmean(ers_dif))
        # ers_l.append(ers_dif)

    ers_sn = np.array(ers_l)
    return ers_sn


def ERS_all_sn():
    atlas = get_atlas(combine_regions=False, bilateral=False)
    # print(atlas['ROIs'])
    # quit()
    age2sn = get_all_sns(ret=True)
    ers_l_all = []
    sns = age2sn[1]
    print(f'{len(sns)=}')
    fps = ['bl2_fMRI', 'obj2_fMRI', 'vis2_fMRI', 'con2_fMRI']
    for i, sn in tqdm(enumerate(sns), desc='ERS, looping subjects'):


        ers_l_all.append(ERS_sn(sn, atlas))
        ers_all = np.array(ers_l_all)
        if i < 3:
            continue
        for j, ROI in enumerate(atlas['ROIs']):
            ers = ers_all[:, j]
            M = np.nanmean(ers)
            SD = np.nanstd(ers)
            N = len(ers[~np.isnan(ers)])
            SE = SD / np.sqrt(N)
            t = M / SE
            p = stats.t.sf(np.abs(t), len(ers)-1)
            print(f'{ROI}, t[{N-1}]={t:.2f}, p={p:.3f}')


if __name__ == '__main__':
    ERS_all_sn()