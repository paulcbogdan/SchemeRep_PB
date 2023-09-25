from tqdm import tqdm

from PFC_analysis import load_and_get_ROI_vecs
from ROIs import get_BN_and_resample, get_atlas
from organize_bhv import get_all_sns, get_trial_info
from single_trial_conn import corr_last_dim
import numpy as np
import scipy.stats as stats

def ERS_sn(sn, atlas):
    df_sn = get_trial_info(sn)
    ROI2vecs_enc, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='con_fMRI')
    ROI2vecs_ret, _ = load_and_get_ROI_vecs(sn, atlas,
                                            fp_fMRI_col='scn_fMRI')
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
        # print(f'{hits=}')
        # print(len(ers))
        # print()
        ers_l.append(np.nanmean(ers[hits]))
    ers_sn = np.array(ers_l)
    return ers_sn


def ERS_all_sn():
    atlas = get_atlas(combine_regions=True, bilateral=False)
    # print(atlas['ROIs'])
    # quit()
    age2sn = get_all_sns(ret=True)
    ers_l_all = []
    for i, sn in tqdm(enumerate(age2sn[1]), desc='ERS'):
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