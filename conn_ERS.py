import numpy as np

from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial, prep_for_pairwise


def ERS_ROI_pairwise(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc',
           combine_regions=False):
    df_sn = get_trial_info(sn)
    df_sn.sort_values('obj', inplace=True)
    ROI2vecs_enc, ROI2vecs_ret = get_ROI_vecs_wrap(sn, atlas, fp0, df_sn,
                                                   fp1=fp1, networks=False,
                                                   org_by_region=False,
                                                   cross_region=False,
                                                   conn=conn,
                                                   combine_regions=False,)

    ROI2vecs_enc, region_order, score_ar, IRAFs_ar = \
        prep_for_pairwise(ROI2vecs_enc, atlas)
    ROI2vecs_ret, _, _, _ = prep_for_pairwise(ROI2vecs_ret, atlas)

    # score_ar = np.full((len(ROI2vecs_enc), len(ROI2vecs_enc)), np.nan)
    # IRAFs_ar = np.full((len(ROI2vecs_enc), len(ROI2vecs_enc), 114), np.nan)

    for i, ROI0 in enumerate(region_order):
        vecs0_enc = ROI2vecs_enc[ROI0]
        vecs0_ret = ROI2vecs_ret[ROI0]
        keeps = np.logical_and(~np.isnan(vecs0_enc).any(axis=0),
                               ~np.isnan(vecs0_ret).any(axis=0))
        vecs0_enc = vecs0_enc[:, keeps]
        vecs0_ret = vecs0_ret[:, keeps]

        for j, ROI1 in enumerate(region_order):
            if ROI1 > ROI0:
                continue
            elif ROI0 == ROI1:
                vecs_enc = get_conn_vecs(vecs0_enc, conn=conn)
                vecs_ret = get_conn_vecs(vecs0_ret, conn=conn)
            else:
                vecs1_enc = ROI2vecs_enc[ROI1]
                vecs1_ret = ROI2vecs_ret[ROI1]
                keeps = np.logical_and(~np.isnan(vecs1_enc).any(axis=0),
                                       ~np.isnan(vecs1_ret).any(axis=0))
                vecs1_enc = vecs1_enc[:, keeps]
                vecs1_ret = vecs1_ret[:, keeps]
                vecs_enc = get_conn_vecs(vecs0_enc, vecs1_enc, conn=conn)
                vecs_ret = get_conn_vecs(vecs0_ret, vecs1_ret, conn=conn)

            assert vecs_enc.shape == vecs_ret.shape
            if vecs_enc.shape[1] == 1:
                score_ar[i, j] = np.nan
                score_ar[j, i] = np.nan
                IRAFs_ar[i, j, :] = np.full((114), np.nan)
                IRAFs_ar[j, i, :] = np.full((114), np.nan)
                # print(f'Only one edge: {ROI0}, {ROI1}')
                continue

            ERS_ar = get_trial_x_trial(vecs_enc, vecs_ret,
                                       trial_similarity=trial_similarity)

            ERS_sames = np.diag(ERS_ar)
            ERS_ar_ = ERS_ar.copy()
            ERS_ar_[np.eye(len(ERS_ar), dtype=bool)] = np.nan
            ERS_elses = np.nanmean(ERS_ar_, axis=1)

            ERS_dif = ERS_sames - ERS_elses
            score = np.nanmean(ERS_dif)

            score_ar[i, j] = score
            score_ar[j, i] = score
            IRAFs_ar[i, j, :] = ERS_dif
            IRAFs_ar[j, i, :] = ERS_dif

    return score_ar, np.nan, IRAFs_ar


def ERS_sn(sn, atlas, fp0 = 'bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc',
           combine_regions=False):
    BOLD = conn == 'BOLD'
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn, easy_override=True)
    # df_sn.sort_values('obj_trial', inplace=True)
    sess = fp0.split('_')[0].replace('2', '').replace('3', '').replace('4', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True) # added to help with runw-wise sorting

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
