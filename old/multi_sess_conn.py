import numpy as np
import scipy.stats as stats

from atlas_utils import get_atlas
from fMRI_proc import get_ROI_vecs
from organize_bhv import get_trial_info
from old.plot_gen import plot_connectivity
from connRSA.single_trial_conn import corr_matrix_last_two_dim
from stim import get_stim_RDM, get_semantic_vectors, get_DNN_vecs
from old.modularity_testing import get_partition_matrix
from utils import tril_flat, stdize

def RSA_trials():
    # trials_re = trials_matrices.reshape(trials_matrices.shape[0], -1)
    # trials_re = trials_re.T
    # trials_re_ = trials_re[:, None, :]
    # trials_re = trials_re[:, :, None]
    #
    # RSM_trials_re = abs(trials_re - trials_re_)
    # RSM_trials_re_flat = RSM_trials_re[:, trils[0], trils[1]]
    # RSM_trials_re_flat = stdize(RSM_trials_re_flat, axis=1)
    # trials_r_RSA = np.nanmean(RSM_trials_re_flat * RSM_stim_flat, axis=1)
    # trials_r_RSA = trials_r_RSA.reshape(len(atlas['ROIs']), len(atlas['ROIs']))
    pass

def do_multi_sess_conn_sn(sn, atlas, d_vecs):
    df_sn = get_trial_info(sn, easy_override=True)

    fp2ROI2vecs = {}
    fp_fMRI_cols = ['bl2_fMRI']#, 'con2_fMRI']
    ar_all_sess = []
    for fp_fMRI_col in fp_fMRI_cols:
        ROI2vecs = get_ROI_vecs(sn, atlas, fp_fMRI_col, df_sn, nan_thresh=1.0)
        fp2ROI2vecs[fp_fMRI_col] = ROI2vecs # shape will vary between sessions
        ar = []
        for ROI, vecs in ROI2vecs.items():
            # ROI2vecs[ROI] = np.nanmean(vecs, axis=0)
            ar.append(np.nanmean(vecs, axis=1))
            # print(f'Test: {np.nanmean(vecs, axis=1).shape}')
        ar_all_sess.append(ar)
        # print(np.array(ar).shape)
        continue
        # fp2ROI2vecs[fp_fMRI_col] = np.array(ar)
        # print(fp2ROI2vecs[fp_fMRI_col].shape)
        # print(ROI2vecs['1 SFG_L_7_1'].shape)
    ar_all_sess = np.array(ar_all_sess)
    ar_all_sess = np.transpose(ar_all_sess, (2, 1, 0))
    ar_all_sess = stdize(ar_all_sess, axis=0)

    RSM_stim = get_stim_RDM(df_sn, d_vecs, obj_only=True, take_abs=False) # was previously add=True?!?!
    trils = np.tril_indices(114, k=-1)
    RSM_stim_flat = RSM_stim[trils]
    RSM_stim_flat = stdize(RSM_stim_flat)

    trials_matrices, trials_flats = corr_matrix_last_two_dim(ar_all_sess,
                                                             euc_dist=True,
                                                             )

    ROI_regions = atlas['ROI_regions']
    idxs_occ = [i for i, region in enumerate(ROI_regions) if
                #('EVC' in region) or
                ('LOC' in region)#('sOcG' in region) or
                #('FuG' in region)
                ]
    trials_occ_matrices = get_partition_matrix(trials_matrices, idxs_occ)
    trials_flats = tril_flat(trials_occ_matrices)
    trials_flats = trials_flats[:, ~np.any(np.isnan(trials_flats), axis=0)]

    RSM_fMRI, _ = corr_matrix_last_two_dim(trials_flats, euc_dist=True)
    # RSM_fMRI = np.corrcoef(trials_flats)
    trils = np.tril_indices_from(RSM_fMRI, k=-1)
    RSM_fMRI_flat = RSM_fMRI[trils]

    r, p = stats.spearmanr(RSM_fMRI_flat, RSM_stim_flat)
    print(f'{sn}, corr: {r=:.3f}')
    trials_r_RSA = 0
    return r, trials_r_RSA



def do_multi_sess_conn(semantic=True):
    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111',
           '112', '113', '114', '115', '117', '118', '119', '120', '123', '124',
           '126', '127', '128', '129', '130', '131', '132', '134', '135', '136',
           '137']
    atlas = get_atlas(combine_regions=False, combine_bilateral=False)
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)

    rs = []
    all_edge_RSMs = []
    for sn in sns[::-1]:
        try:
            r, trials_r_RSA = do_multi_sess_conn_sn(sn, atlas, d_vecs)
            all_edge_RSMs.append(trials_r_RSA)
            rs.append(r)
            m = np.mean(rs)
            sd = np.std(rs)
            se = sd / np.sqrt(len(rs))
            t = m / se
            print(f'Group-level: {m=:.3f}, {sd=:.3f}, {se=:.3f}, {t=:.3f}')
            M_edge_RSM = np.nanmean(all_edge_RSMs, axis=0)
            SD_edge_RSM = np.nanstd(all_edge_RSMs, axis=0)
            SE_edge_RSM = SD_edge_RSM / np.sqrt(len(all_edge_RSMs))
            t_edge_RSM = M_edge_RSM / SE_edge_RSM
            t_edge_RSM[abs(t_edge_RSM) < 2] = np.nan
            n = len(all_edge_RSMs)
            plot_connectivity(t_edge_RSM,
                              atlas['ticks'], atlas['tick_labels'],
                              atlas['tick_lows'], no_avg=True,
                              vmin=-4, vmax=4,
                              xlabel='', ylabel='',
                              title=f'Subject: {sn}, deconvolved trial TRs ({n=})')
        except Exception as e:
            print(f'Error ({sn}): {e}')

if __name__ == '__main__':
    do_multi_sess_conn()








