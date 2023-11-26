import numpy as np

from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial


def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc',
           combine_regions=False):
    BOLD = conn == 'BOLD'
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn)
    ROI2vecs_enc, ROI2vecs_ret = get_ROI_vecs_wrap(sn, atlas, fp0, df_sn,
                                                   fp1=fp1, networks=networks,
                                                   org_by_region=not BOLD,
                                                   cross_region=cross_region,
                                                   conn=conn,
                                                   combine_regions=combine_regions,)
    scores = []
    sizes = []
    scores_by_trial = []
    for ROI in ROI2vecs_enc.keys():
        try:
            vecs_enc_BOLD = ROI2vecs_enc[ROI]
            vecs_ret_BOLD = ROI2vecs_ret[ROI]
        except KeyError:
            scores.append(np.nan)
            continue
        assert vecs_enc_BOLD.shape == vecs_ret_BOLD.shape
        keeps = np.logical_and(~np.isnan(vecs_enc_BOLD).any(axis=0),
                               ~np.isnan(vecs_ret_BOLD).any(axis=0))
        vecs_enc_BOLD = vecs_enc_BOLD[:, keeps]
        vecs_ret_BOLD = vecs_ret_BOLD[:, keeps]

        sizes.append(np.sum(keeps))
        if BOLD or cross_region:
            vecs_enc = vecs_enc_BOLD
            vecs_ret = vecs_ret_BOLD
        else:
            vecs_enc = get_conn_vecs(vecs_enc_BOLD, conn=conn)
            vecs_ret = get_conn_vecs(vecs_ret_BOLD, conn=conn)

        ERS_ar = get_trial_x_trial(vecs_enc, vecs_ret,
                                   trial_similarity=trial_similarity)

        ERS_sames = np.diag(ERS_ar)
        ERS_ar_ = ERS_ar.copy()
        ERS_ar_[np.eye(len(ERS_ar), dtype=bool)] = np.nan
        ERS_elses = np.nanmean(ERS_ar_, axis=1)

        ERS_dif = ERS_sames - ERS_elses
        score = np.nanmean(ERS_dif)
        scores.append(score)
        scores_by_trial.append(ERS_dif)

    scores = np.array(scores)
    return scores, sizes, scores_by_trial
