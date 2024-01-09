import random
from collections import defaultdict
from itertools import combinations

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as stats

from atlas_utils import get_atlas
from conn_utils import get_BNA_ROIs
from fMRI_proc import get_ROI_vecs
from modularity import get_partition_matrix
from organize_bhv import get_trial_info
from org_sns import get_all_sns
from utils import stdize
import pickle

def load_FC_for_Lifu(atlas_name='BNA', fp='obj3_fMRI', split=False, key='inc',
                     key_vals=(1, 2, 3), odd_even=False, pad_nan=False,
                     do_sort=False, fp_all=False, voxelwise=False,
                     regionwise=False):
    if atlas_name == 'schaefer':
        atlas = get_atlas(schaefer=True)
    else:
        atlas = get_atlas(combine_regions=False,
                          combine_bilateral=False,
                          split=split, split_code='xyz',
                          new_space='3' in fp or '4' in fp)
    coords = atlas['coords']
    # with open(f'cache/{atlas_name}_coords.pkl', 'wb') as f:
    #     pickle.dump(coords, f)
    # quit()
    age2sn = get_all_sns('all' if fp_all else fp, sh=False)
    # age_sn_inc_conn = []
    # age_sn_conn = []
    sn_inc_conn = []
    sn_inc_activity = []
    sn_conn = []

    # age_l = []
    age2idxs = defaultdict(list)
    sn_idx = 0
    ROI2act = defaultdict(list)
    Y = []
    grp_idxs = []
    for i, age in enumerate([1, 2]):
        sns = age2sn[age]
        print(f'{len(sns)=}')
        # if atlas == 'schaefer':
        #     ROIs_l = get_BNA_ROIs(code='schaefer')
        # else:
        ROIs_l = get_BNA_ROIs(code=atlas_name)

        for sn in sns:
            print(f'Prepping FC: {sn=}')
            print(sn_idx)
            df_sn = get_trial_info(sn, easy_override=True)
            if do_sort:
                sess = fp.split('_')[0].replace('2', '').replace('3', '').\
                    replace('4', '')
                df_sn.sort_values(by=f'{sess}_trial', inplace=True)
            ROI2vecs0 = get_ROI_vecs(sn, atlas, fp, df_sn, nan_thresh=1.01,
                                     drop_nan_voxels=False,
                                     org_by_region=regionwise,
                                     easy_override=True if sn == '132'
                                                        else False,
                                     combine_regions=False)
            print(list(ROI2vecs0))
            if voxelwise or regionwise:
                for ROI in ROI2vecs0:
                # for ROI in ROIs_l:
                    ROI2act[ROI].append(ROI2vecs0[ROI])
                y = []
                for v in df_sn[key]:
                    for i, key_val in enumerate(key_vals):
                        if v == key_val:
                            y.append(i)
                            break
                    else:
                        y.append(np.nan)
                Y.extend(y)
                grp_idxs.extend([sn_idx]*114)
                age2idxs[age].append(sn_idx)
                sn_idx += 1
                continue

            activity_ar = []
            for ROI in ROIs_l:
                if ROI not in ROI2vecs0:
                    activity_ar.append(np.full(114, np.nan))
                else:
                    activity_ar.append(np.nanmean(ROI2vecs0[ROI], axis=1))
            activity_ar = np.array(activity_ar)
            conn_no_cond = np.corrcoef(activity_ar)
            sn_conn.append(conn_no_cond)
            activity_inc = []
            conns = []
            for i, inc in enumerate(key_vals):
                if key == 'rand':
                    if key_vals == (1, 2, 3):
                        matching_trials = df_sn['inc'].sample(frac=1.) == inc
                    else:
                        matching_trials = df_sn['vis_hit'].sample(frac=1.) == inc
                else:
                    matching_trials = df_sn[key] == inc
                if odd_even:
                    matching_trials_even = []
                    matching_trials_odd = []
                    cnt = 0
                    org = [False] * (sum(matching_trials) // 2) + \
                          [True] * (sum(matching_trials) // 2)
                    if sum(matching_trials) % 2:
                        org += [False]
                    random.shuffle(org)
                    for trial in matching_trials:
                        if trial:
                            matching_trials_even.append(org[cnt])
                            matching_trials_odd.append(not org[cnt])
                            cnt += 1
                        else:
                            matching_trials_even.append(False)
                            matching_trials_odd.append(False)
                    activity_inc.append([activity_ar[:, matching_trials_even],
                                            activity_ar[:, matching_trials_odd]])

                    conns.append([np.corrcoef(activity_ar[:, matching_trials_even]),
                                  np.corrcoef(activity_ar[:, matching_trials_odd])])
                else:
                    activity_ar_matched = np.full(activity_ar.shape, np.nan)
                    activity_ar_matched[:, matching_trials] = \
                        activity_ar[:, matching_trials]
                    activity_inc.append(activity_ar_matched)

                    conn_inc = np.corrcoef(activity_ar[:, matching_trials])
                    conns.append(conn_inc)
            conns = np.array(conns)
            sn_inc_conn.append(conns)
            sn_inc_activity.append(activity_inc)
            age2idxs[age].append(sn_idx)
            sn_idx += 1

    if voxelwise:
        for ROI, vals in ROI2act.items():
            ROI2act[ROI] = np.array(vals)
            print(f'{ROI}, {ROI2act[ROI].shape=}')
        return ROI2act, Y, grp_idxs, age2idxs
    else:
        sn_inc_conn = np.array(sn_inc_conn)
        sn_conn = np.array(sn_conn)
        diag = np.diag_indices(sn_inc_conn.shape[-1])
        sn_inc_conn[..., diag[0], diag[1]] = np.nan
        sn_conn[..., diag[0], diag[1]] = np.nan
        sn_inc_activity = np.array(sn_inc_activity)
        return sn_inc_conn, sn_conn, age2idxs, sn_inc_activity


def get_ylim_settings(sn_inc_conn, age2idxs, ps, do_division=True):
    y_low = 1e6
    y_high = -1e6
    for p in ps:
        for age in [1, 2]:
            sn_idxs = age2idxs[age]
            age_sn_inc_conn = sn_inc_conn[sn_idxs]
            p_mat = get_partition_matrix(age_sn_inc_conn, p)
            p_M_within_connectivity = np.nanmean(p_mat, axis=(-2, -1))
            between_mask = np.full((246, 246), False)
            between_mask[p, :] = True
            between_mask[:, p] = True
            between_mask[np.ix_(p, p)] = False
            p_between_edges = age_sn_inc_conn[:, :, between_mask]
            p_M_between_connectivity = np.nanmean(p_between_edges, axis=-1)
            if do_division:
                integration = p_M_between_connectivity / p_M_within_connectivity
            else:
                integration = p_M_between_connectivity
            idxs = integration.shape[1]
            for i in range(idxs):
                M_i = np.mean(integration[:, i])
                SD_i = np.std(integration[:, i])
                SE = SD_i / np.sqrt(integration.shape[0])
                candidate_high = M_i + SE * 2
                candidate_low = M_i - SE * 2
                if candidate_high > y_high:
                    y_high = candidate_high
                if candidate_low < y_low:
                    y_low = candidate_low
    return y_low, y_high


def calculate_within_between(sn_inc_conn, age2idxs, p, labels,
                             y_low, y_high,
                             do_division=True, suptitle=None,
                             ):

    matplotlib.rc('font', **{'size': 14})
    fig, axs = plt.subplots(1, 2)
    Ms_all = []
    SDs_all = []
    SEs_all = []
    for age in [1, 2]:
        plt.sca(axs[age-1])
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        p_mat = get_partition_matrix(age_sn_inc_conn, p)
        p_M_within_connectivity = np.nanmean(p_mat, axis=(-2, -1))
        between_mask = np.full((246, 246), False)
        between_mask[p, :] = True
        between_mask[:, p] = True
        between_mask[np.ix_(p, p)] = False
        p_between_edges = age_sn_inc_conn[:, :, between_mask]
        p_M_between_connectivity = np.nanmean(p_between_edges, axis=-1)

        if do_division:
            integration = p_M_between_connectivity / p_M_within_connectivity
            # plt.ylim((0.7, 1.1))
            plt.ylabel('Integration')
        else:
            integration = p_M_between_connectivity
            # plt.ylim((0.24, 0.31))
            plt.ylabel('Between connectivity')

        Ms = []
        errs = []

        print(f'\tAge: {age}')
        idxs = integration.shape[1]
        for i in range(idxs):
            M_i = np.mean(integration[:, i])
            SD_i = np.std(integration[:, i])
            SE = SD_i / np.sqrt(integration.shape[0])
            Ms.append(M_i)
            errs.append(SE)
            print(f'Mean  {i}: {M_i:.3f} [{SE:.3f}]')

        plt.bar(labels, Ms, yerr=errs, color='dodgerblue' if age == 1 else 'r',
                capsize=6)
        dist = (np.max(Ms) - np.min(Ms) + 2 * np.max(errs)) / 2
        mid = (np.max(Ms) + np.min(Ms)) / 2

        comparisons = combinations(list(range(idxs)), 2)
        height = mid + dist * 1.05
        for (i, j) in comparisons:
            dif = integration[:, i] - integration[:, j]
            M_dif = np.mean(dif)
            SD_dif = np.std(dif)
            SE = SD_dif / np.sqrt(integration.shape[0])
            t = M_dif / SE
            p_val = stats.t.sf(np.abs(t), integration.shape[0]-1)*2
            if p_val < 0.1:
                plt.plot([i, j], [height, height], color='k')
                plt.text((i+j)/2, height + dist * 0.02,
                         f'p = {p_val:.3f}', ha='center', va='bottom')
                height += dist * 0.35

            print(f't-test {i} - {j}: {M_dif:.3f} [{SE:.3f}], {t=:.3f}')
        Y_low = mid - dist * 1.3
        Y_high = height + dist * 0.15


        all_inc = np.mean(integration, axis=1)
        M_inc = np.mean(all_inc)
        SD_inc = np.std(all_inc)
        SE_inc = SD_inc / np.sqrt(integration.shape[0])
        print(f'All inc: {M_inc:.3f} [{SE_inc:.3f}]')
        Ms_all.append(M_inc)
        SDs_all.append(SD_inc)
        SEs_all.append(SE_inc)

        plt.ylim((Y_low, Y_high))
        age2str = {1: 'YA', 2: 'OA'}
        # matplotlib.use('ps')
        # matplotlib.rc('text', usetex=True)
        # matplotlib.rc('text.latex', preamble=r'\usepackage{color}')
        # plt.title(r'\textcolor{red}{Today}')
        plt.title(f'Age = {age2str[age]}', color='dodgerblue' if age == 1 else 'r')
        plt.ylim((y_low, y_high))

    plt.tight_layout()
    if suptitle is not None: plt.suptitle(suptitle)
    plt.show()

    plt.figure(figsize=(4, 4))
    t, p_val = stats.ttest_ind_from_stats(Ms_all[0], SDs_all[0], len(age2idxs[1]),
                                      Ms_all[1], SDs_all[1], len(age2idxs[2]))
    print(f'Between age t-test: {t=:.3f}, {p_val=:.3f}')
    dist = (np.max(Ms_all) - np.min(Ms_all) + 2 * np.max(SEs_all)) / 2
    mid = (np.max(Ms_all) + np.min(Ms_all)) / 2
    height = mid + dist * 1.1
    plt.plot([0, 1], [height, height], color='k')
    plt.text(0.5, height + dist * 0.05,
             f'p = {p_val:.3f}', ha='center', va='bottom')
    Y_low = mid - dist * 1.3
    Y_high = height + dist * 0.35
    # plt.ylim((Y_low, Y_high))
    plt.title(suptitle)
    plt.bar(['Young', 'Old'], Ms_all, yerr=SEs_all, color=['dodgerblue', 'r'],
            capsize=6)
    if do_division:
        # plt.ylim((0.7, 1.1))
        plt.ylabel('Integration')
    else:
        # plt.ylim((0.24, 0.31))
        plt.ylabel('Between connectivity')
    plt.ylim((y_low, y_high))
    plt.tight_layout()
    plt.show()


def reconfiguration(sn_inc_conn, age2idxs, p, title=None,
                    col0=0, col1=1, plot=False):
    matplotlib.rc('font', **{'size': 14})
    Ms = []
    SEs = []
    labels =[ 'Young', 'Old']
    plt.figure(figsize=(4, 4))
    rs_both = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        p_mat = get_partition_matrix(age_sn_inc_conn, p)
        rs = []
        for sn_i in range(p_mat.shape[0]):
            R_mat = p_mat[sn_i, col0]
            F_mat = p_mat[sn_i, col1]
            tril = np.tril_indices_from(R_mat, k=-1)
            R_flat = R_mat[tril]
            F_flat = F_mat[tril]
            r, _ = stats.spearmanr(R_flat, F_flat, nan_policy='omit')

            rs.append(1-r)

        # for sn_i in range(p_mat.shape[0]):
        #     R_mat = p_mat[sn_i, 0]
        #     F_mat = p_mat[sn_i, 1]
        #     K_mat = p_mat[sn_i, 2]
        #     tril = np.tril_indices_from(R_mat, k=-1)
        #     R_flat = R_mat[tril]
        #     F_flat = F_mat[tril]
        #     K_flat = K_mat[tril]
        #     r, _ = stats.spearmanr(R_flat, F_flat, nan_policy='omit')
        #     r2, _ = stats.spearmanr(R_flat, K_flat, nan_policy='omit')
        #     r3, _ = stats.spearmanr(F_flat, K_flat, nan_policy='omit')
        #     # rs.append(1-r)
        #     rs.append(3-r-r2-r3)

        M_reconfig = np.mean(rs)
        SD_reconfig = np.std(rs)
        SE_reconfig = SD_reconfig / np.sqrt(len(rs))
        print(f'{age=} Reconfiguration ({title}): {M_reconfig:.3f} '
              f'[{SE_reconfig:.3f}]')
        Ms.append(M_reconfig)
        SEs.append(SE_reconfig)
        rs_both.append(rs)
    super_M = np.mean(rs_both[0] + rs_both[1])
    super_SD = np.std(rs_both[0] + rs_both[1])
    rs_both[0] = np.array(rs_both[0])
    # rs_both[0] = (rs_both[0] - super_M) / super_SD
    rs_both[1] = np.array(rs_both[1])
    dif = np.mean(rs_both[0]) - np.mean(rs_both[1])
    # rs_both[1] = (rs_both[1] - super_M) / super_SD
    M0 = np.mean(rs_both[0])
    M1 = np.mean(rs_both[1])
    Ms = [M0, M1]
    SEs = [np.std(rs_both[0]) / np.sqrt(len(rs_both[0])),
           np.std(rs_both[1]) / np.sqrt(len(rs_both[1]))]

    t, p_val = stats.ttest_ind(rs_both[0], rs_both[1])
    d = t / np.sqrt(len(rs_both[0]) + len(rs_both[1]))
    print(f'\tAge-related reconfiguration ({title}): {t=:.3f}, {p_val=:.3f}, '
          f'{d=:.3f}, {dif=:.3f}')
    dist = (np.max(Ms) - np.min(Ms) + 2 * np.max(SEs)) / 2
    mid = (np.max(Ms) + np.min(Ms)) / 2
    height = mid + dist * 1.05
    if plot:
        return M0, M1, d, dif

    plt.plot([0, 1], [height, height], color='k')
    plt.text(0.5, height + dist * 0.02,
             f'p = {p_val:.3f}', ha='center', va='bottom')
    height += dist * 0.35
    plt.bar(labels, Ms, yerr=SEs, color=['dodgerblue', 'r'], capsize=6)
    Y_low = mid - dist * 1.3
    Y_high = height + dist * 0.15
    plt.ylim((Y_low, Y_high))
    # plt.ylim((0.6, 0.8))
    if 'vs' in title:
        plt.ylabel('Schema-related reconfiguration')
    else:
        plt.ylabel('Memory-related reconfiguration')
    if title is not None: plt.title(title)
    plt.tight_layout()
    plt.show()
    return M0, M1, d, dif


def analyze_subject_specific(sn_inc_conn, age2idxs):
    sames_all = []
    difs_all = []
    efs_all = []
    n_elements = sn_inc_conn.shape[1] * sn_inc_conn.shape[2]
    # print(M_conn.shape)
    # quit()
    # sn_inc_conn = sn_inc_conn - M_conn[:, None, None, :, :]
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_conn = sn_inc_conn[sn_idxs]
        M_conn = np.nanmean(age_sn_inc_conn, axis=(1, 2))

        sames = []
        difs = []
        efs = []

        for sn in range(age_sn_inc_conn.shape[0]):
            sames_sn = []
            difs_sn = []
            for inc0 in range(age_sn_inc_conn.shape[1]):
                for inc1 in range(age_sn_inc_conn.shape[1]):
                    if inc0 > inc1:
                        continue

                    mat0 = age_sn_inc_conn[sn, inc0, 0]
                    mat1 = age_sn_inc_conn[sn, inc1, 1]
                    mat_else = M_conn[sn]
                    mat_else = (mat_else * n_elements - mat0 - mat1) / (n_elements - 2)
                    # mat0 -= mat_else
                    # mat1 -= mat_else
                    # mat0 -= (mat_else * n_elements - mat0) / (n_elements - 1)
                    # mat1 -= (mat_else * n_elements - mat1) / (n_elements - 1)
                    trils = np.tril_indices_from(mat0, k=-1)
                    flat0_a = mat0[trils]
                    flat1_a = mat1[trils]
                    # r_a = np.nanmean(flat0_a - flat1_a)
                    r_a, _ = stats.spearmanr(flat0_a, flat1_a, nan_policy='omit')

                    if inc0 == inc1:
                        sames_sn.append(r_a)
                    else:
                        mat0 = age_sn_inc_conn[sn, inc0, 1]
                        mat1 = age_sn_inc_conn[sn, inc1, 0]
                        mat_else = M_conn[sn]
                        # mat0 -= (mat_else * n_elements - mat0) / (n_elements - 1)
                        # mat1 -= (mat_else * n_elements - mat1) / (n_elements - 1)

                        # plt.imshow(mat0)
                        # plt.show()
                        # mat_else = (mat_else * n_elements - mat0 - mat1) / (n_elements - 2)
                        # mat0 -= mat_else
                        # mat1 -= mat_else

                        # if sn == 0 and inc0 == 0:
                        #     plt.imshow(mat_else)
                        #     plt.title(f'age = {age}')
                        #     plt.show()

                        # plt.imshow(mat0)
                        # plt.show()
                        # quit()

                        trils = np.tril_indices_from(mat0, k=-1)
                        flat0_b = mat0[trils]
                        flat1_b = mat1[trils]
                        r_b, _ = stats.spearmanr(flat0_b, flat1_b, nan_policy='omit')
                        # r_b = np.nanmean(flat0_b - flat1_b)
                        difs_sn.extend([r_a, r_b])
                        # mat0 = age_sn_inc_conn[sn, inc0, 0]
                        # mat1 = age_sn_inc_conn[sn, inc1, 0]
                        # trils = np.tril_indices_from(mat0, k=-1)
                        # flat0_a = mat0[trils]
                        # flat1_a = mat1[trils]
                        # r_a, _ = stats.spearmanr(flat0_a, flat1_a, nan_policy='omit')
                        # mat0 = age_sn_inc_conn[sn, inc0, 1]
                        # mat1 = age_sn_inc_conn[sn, inc1, 1]
                        # trils = np.tril_indices_from(mat0, k=-1)
                        # flat0_b = mat0[trils]
                        # flat1_b = mat1[trils]
                        # r_b, _ = stats.spearmanr(flat0_b, flat1_b, nan_policy='omit')
                        # difs_sn.extend([r_a, r_b])
            if sn == 0:
                print(f'{len(sames_sn)=} {len(difs_sn)=}')
            M_same = np.mean(sames_sn)
            M_dif = np.mean(difs_sn)
            sames.append(M_same)
            difs.append(M_dif)
            efs.append(M_same - M_dif)
        M_age_same = np.mean(sames)
        M_age_dif = np.mean(difs)
        M_age_ef = np.mean(efs)
        SD_age_ef = np.std(efs)
        SE_age_ef = SD_age_ef / np.sqrt(len(efs))
        t_age_ef = M_age_ef / SE_age_ef
        p_age_ef = stats.t.sf(np.abs(t_age_ef), len(efs)-1)*2
        print(f'Subject specific effect ({age}) | same: {M_age_same:.3f}, '
              f'dif: {M_age_dif:.3f}, t = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        sames_all.append(sames)
        difs_all.append(difs)
        efs_all.append(efs)
    t, p = stats.ttest_ind(efs_all[0], efs_all[1])
    print(f'\tAge x Effect: t = {t:.3f}, p = {p:.3f}')


def subj_specific_repeated(sn_inc_activity, age2idxs, top_edges_mat,
                           p,
                           variability=True):
    # Randomly split each condition into 2 sets. submit to the odd/even analysis
    efs_all = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        efs_age = []
        sames_age = []
        difs_age = []
        for sn in range(age_sn_inc_act.shape[0]):
            efs = []
            sames = []
            difs = []
            smaller_trial = 1e10
            for inc in range(age_sn_inc_act.shape[1]):
                act = age_sn_inc_act[sn, inc]
                nans = np.all(np.isnan(act), axis=0)
                act = act[:, ~nans]
                n_trials = act.shape[1]
                if n_trials < smaller_trial:
                    smaller_trial = n_trials

            for shuffle in range(100):
                inc_shuffled_conns = []
                for inc in range(age_sn_inc_act.shape[1]):
                    act = age_sn_inc_act[sn, inc]
                    nans = np.all(np.isnan(act), axis=0)
                    act = act[:, ~nans]
                    n_trials = act.shape[1]
                    # print(f'{n_trials=}, {smaller_trial=}')
                    # smaller_trial = n_trials
                    matcher_pre = [True] * (smaller_trial // 2) + \
                                  [False] * (smaller_trial // 2) + \
                                  [np.nan] * (n_trials - smaller_trial)
                    if smaller_trial % 2:
                        matcher_pre += [np.nan]
                    # print(len(matcher_pre))
                    random.shuffle(matcher_pre)
                    matcher0 = []
                    matcher1 = []
                    for i, b in enumerate(matcher_pre):
                        if b is np.nan:
                            matcher0.append(False)
                            matcher1.append(False)
                        elif b:
                            matcher0.append(True)
                            matcher1.append(False)
                        else:
                            matcher0.append(False)
                            matcher1.append(True)

                    # matcher = [True] * (smaller_trial // 2)

                    n_trials = act.shape[1]
                    # n_trials = smaller_trial
                    # matcher = [False] * (n_trials // 2) + \
                    #           [True] * (n_trials // 2)
                    # if n_trials % 2:
                    #     matcher += [False]
                    # matcher = np.array(matcher)
                    # activity0 = act[:, matcher]
                    # print(act.shape[1])
                    activity0 = act[:, matcher0]
                    # print(activity0.shape)
                    # plt.imshow(activity0)
                    # plt.show()
                    # quit()
                    # activity1 = act[:, ~matcher]
                    activity1 = act[:, matcher1]
                    conn0 = np.corrcoef(activity0)
                    conn0[~top_edges_mat] = np.nan
                    # conn0 = get_partition_matrix(conn0, p)
                    conn1 = np.corrcoef(activity1)
                    conn1[~top_edges_mat] = np.nan
                    # conn1 = get_partition_matrix(conn1, p)

                    tril = np.tril_indices(conn0.shape[0], k=-1)
                    flat0 = conn0[tril]
                    flat1 = conn1[tril]
                    nans = np.isnan(flat0) | np.isnan(flat1)
                    flat0 = flat0[~nans]
                    flat1 = flat1[~nans]
                    inc_shuffled_conns.append([flat0, flat1])
                if variability:
                    if age_sn_inc_act.shape[1] == 2:
                        # var0 = np.mean(abs(inc_shuffled_conns[0][0] -
                        #                inc_shuffled_conns[0][1]))
                        # var1 = np.mean(abs(inc_shuffled_conns[1][0] -
                        #                inc_shuffled_conns[1][1]))
                        var0, _ = stats.spearmanr(inc_shuffled_conns[0][0],
                                                  inc_shuffled_conns[0][1])
                        var1, _ = stats.spearmanr(inc_shuffled_conns[1][0],
                                                    inc_shuffled_conns[1][1])
                        ef = var0 - var1
                        efs.append(ef)
                        sames.append(var0)
                        difs.append(var1)
                else:
                    if age_sn_inc_act.shape[1] == 2:
                        inc0_same, _ = stats.spearmanr(inc_shuffled_conns[0][0],
                                                    inc_shuffled_conns[0][1])
                        inc1_same, _ = stats.spearmanr(inc_shuffled_conns[1][0],
                                                    inc_shuffled_conns[1][1])
                        inc01_a, _ = stats.spearmanr(inc_shuffled_conns[0][0],
                                                  inc_shuffled_conns[1][1])
                        inc01_b, _ = stats.spearmanr(inc_shuffled_conns[1][0],
                                                  inc_shuffled_conns[0][1],)

                        # inc0_same = np.mean(abs(inc_shuffled_conns[0][0] -
                        #                         inc_shuffled_conns[0][1]))
                        # inc1_same = np.mean(abs(inc_shuffled_conns[1][0] -
                        #                         inc_shuffled_conns[1][1]))
                        # inc01_a = np.mean(abs(inc_shuffled_conns[0][0] -
                        #                       inc_shuffled_conns[1][1]))
                        # inc01_b = np.mean(abs(inc_shuffled_conns[1][0] -
                        #                       inc_shuffled_conns[0][1]))

                        ef = (inc0_same + inc1_same - inc01_a - inc01_b) / 2
                        efs.append(ef)
                        sames.append((inc0_same + inc1_same) / 2)
                        difs.append((inc01_a + inc01_b) / 2)
                    else:
                        raise NotImplementedError
            if np.isnan(np.mean(efs)):
                print(f'Nan skip: {age=}, {sn=}')
                continue
            efs_age.append(np.mean(efs))
            sames_age.append(np.mean(sames))
            difs_age.append(np.mean(difs))

        M_age_same = np.mean(sames_age)
        M_age_dif = np.mean(difs_age)
        M_age_ef = np.mean(efs_age)
        SD_age_ef = np.std(efs_age)
        SE_age_ef = SD_age_ef / np.sqrt(len(efs_age))
        t_age_ef = M_age_ef / SE_age_ef
        p_age_ef = stats.t.sf(np.abs(t_age_ef), len(efs_age)-1)*2
        if variability:
            print(f'Variability ({age}) | '
                  f'cond 0: {M_age_same:.3f}, cond 1: {M_age_dif:.3f}, '
                  f't = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        else:
            print(f'Subject specific effect ({age}) | '
                  f'corr same: {M_age_same:.3f}, corr dif: {M_age_dif:.3f}, '
                  f't = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        efs_all.append(efs_age)

    t, p_val = stats.ttest_ind(efs_all[0], efs_all[1])
    print(f'\tAge x Effect: t = {t:.3f}, p = {p_val:.3f}')
