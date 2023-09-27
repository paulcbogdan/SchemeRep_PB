from tqdm import tqdm

from PFC_analysis import load_and_get_ROI_vecs
from ROIs import get_BN_and_resample, get_atlas
from organize_bhv import get_all_sns, get_trial_info
from single_trial_conn import corr_last_dim
import numpy as np
import scipy.stats as stats

from utils import stdize


def ERS_sn(sn, atlas):
    df_sn = get_trial_info(sn)
    ROI2vecs_enc, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='obj_fMRI')
    ROI2vecs_ret, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='con_fMRI')
    ers_l = []
    hits = df_sn['hit_bool'].fillna(False).values
    for ROI in atlas['ROIs']:
        try:
            vecs_enc = ROI2vecs_enc[ROI]
            vecs_ret = ROI2vecs_ret[ROI]
        except KeyError:
            ers_l.append(np.nan)
            continue
        if vecs_enc.shape != vecs_ret.shape:
            ers_l.append(np.nan)
            continue

        ers = corr_last_dim(vecs_enc, vecs_ret)
        # print(ers.shape)
        # quit()
        # print(vecs_ret[:, 0])
        # quit()
        vecs_enc_std = stdize(vecs_enc, axis=1)
        vecs_ret_std = stdize(vecs_ret, axis=1)
        ers_else = np.full(ers.shape, np.nan)
        for i, vec_enc in enumerate(vecs_enc_std):
            vecs_ret_std_ = np.vstack([vecs_ret_std[:i, :],
                                       vecs_ret_std[i+1:, :]])
            prods = vec_enc * vecs_ret_std_
            assert np.sum(np.isnan(prods)) == 0, f'NaNs in prods: {ROI=}, ' \
                                                 f'{np.sum(np.isnan(prods))=}'
            corrs = np.mean(prods, axis=1)
            ers_else[i] = np.mean(corrs)
        ers = ers - ers_else
        # print(ers_else.shape)
        # quit()
        ers_l.append(np.nanmean(ers[hits]))
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