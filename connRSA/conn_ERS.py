from pathlib import Path

import numpy as np

from organize_bhv import get_trial_info
from conn_utils import get_conn_vecs, get_ROI_vecs_wrap, get_trial_x_trial, prep_for_pairwise
from utils import stdize



def ERS_sn(sn, atlas, fp0='bl2_fMRI', fp1='obj2_fMRI',
           networks=True, conn='euc', trial_similarity='euc',
           combine_regions=False, stdize_by_run=False):
    BOLD = 'BOLD' in conn
    cross_region = 'cross_' in conn
    if 'cross_' in conn:
        conn = conn.replace('cross_', '')

    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    # df_sn.sort_values('obj_trial', inplace=True)
    sess = (fp0.split('_')[0].replace('2', '').
            replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True) # added to help with run-wise sorting

    dir_out = fr'cache/conn_RSA/ars/ERS'
    dir_out = fr'{dir_out}/{fp0}_{fp1}_{trial_similarity}_{stdize_by_run}'
    Path(dir_out).mkdir(parents=True, exist_ok=True)

    # org_by_region = (not BOLD) or (networks)
    org_by_region = (not BOLD) or (networks and not combine_regions)
    # org_by_region = (not BOLD) and (networks or not combine_regions)

    # print(org_by_region)
    # quit()

    ROI2vecs_enc, ROI2vecs_ret = get_ROI_vecs_wrap(sn, atlas, fp0, df_sn,
                                                   fp1=fp1, networks=networks,
                                                   org_by_region=org_by_region,
                                                   cross_region=cross_region,
                                                   conn=conn,
                                                   combine_regions=combine_regions,
                                                   easy_override=False)
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
        assert vecs_enc_BOLD.shape == vecs_ret_BOLD.shape, \
            (f'{sn=} ({ROI=}, {fp0}; {fp1}) | {vecs_enc_BOLD.shape=}, '
             f'{vecs_ret_BOLD.shape=}')

        # keep sns missing 1 run
        keeps_enc = np.isnan(vecs_enc_BOLD).sum(axis=0) < 39
        keeps_ret = np.isnan(vecs_ret_BOLD).sum(axis=0) < 39
        keeps = np.logical_and(keeps_enc, keeps_ret)

        # keeps = np.logical_and(~np.isnan(vecs_enc_BOLD).any(axis=0),
        #                        ~np.isnan(vecs_ret_BOLD).any(axis=0))
        vecs_enc_BOLD = vecs_enc_BOLD[:, keeps]
        vecs_ret_BOLD = vecs_ret_BOLD[:, keeps]

        sizes.append(np.sum(keeps))
        if BOLD or cross_region:
            vecs_enc = stdize(vecs_enc_BOLD, axis=0, nans=True,
                          stdize_by_run=stdize_by_run)
            vecs_ret = stdize(vecs_ret_BOLD, axis=0, nans=True,
                          stdize_by_run=stdize_by_run)
            # vecs_enc = vecs_enc_BOLD
            # vecs_ret = vecs_ret_BOLD
        else:
            vecs_enc = get_conn_vecs(vecs_enc_BOLD, conn=conn,
                                     stdize_by_run=stdize_by_run)
            vecs_ret = get_conn_vecs(vecs_ret_BOLD, conn=conn,
                                     stdize_by_run=stdize_by_run)

        if 'avg' in conn:
            vecs_enc = np.nanmean(vecs_enc, axis=-1)[..., None]
            vecs_ret = np.nanmean(vecs_ret, axis=-1)[..., None]

        ERS_ar = get_trial_x_trial(vecs_enc, vecs_ret,
                                   trial_similarity=trial_similarity)

        cmb = '_cmb' if (combine_regions and BOLD) else ''
        fn_matrix = f'{sn}_{ROI}_{conn}{cmb}.npy'
        # print(f'{fn_matrix=}')
        fp_matrix = f'{dir_out}/{fn_matrix}'
        with open(fp_matrix, 'wb') as f:
            np.save(f, ERS_ar)

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
