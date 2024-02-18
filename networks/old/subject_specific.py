import itertools

import networkx as nx
import numpy as np
from numpy import ma as ma
from scipy import stats as stats
from tqdm import tqdm

from connRSA.conn_RSA import get_trial_x_trial_RSM
from connRSA.conn_utils import get_trial_x_trial


def conn_ERS_p(sn_inc_activity0, sn_inc_activity1, age2idxs, edges,
               trial_similarity='corr', activity=False):
    same_Ms_all = []
    dif_Ms_all = []
    out_strs = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        # print(sn_idxs)
        age_sn_inc_act0 = sn_inc_activity0[sn_idxs]
        age_sn_inc_act1 = sn_inc_activity1[sn_idxs]
        same_Ms_age = []
        dif_Ms_age = []

        for sn in range(age_sn_inc_act0.shape[0]):
            sn_inc_act0 = age_sn_inc_act0[sn]
            sn_inc_act1 = age_sn_inc_act1[sn]
            # bad_voxels0 = np.all(np.isnan(sn_inc_act0), axis=(0, 2))
            # bad_voxels1 = np.all(np.isnan(sn_inc_act1), axis=(0, 2))
            # bad_voxels = np.logical_or(bad_voxels0, bad_voxels1)
            # sn_inc_act0 = sn_inc_act0[:, ~bad_voxels, :]
            # sn_inc_act1 = sn_inc_act1[:, ~bad_voxels, :]
            same_Ms = []
            dif_Ms = []
            for inc in range(sn_inc_act0.shape[0]):
                act0 = sn_inc_act0[inc].T
                act1 = sn_inc_act1[inc].T
                if activity:
                    ROIs = [ROI for sl in edges for ROI in sl]
                    ROIs = list(set(ROIs))
                    act0 = act0[:, ROIs]
                    act1 = act1[:, ROIs]
                else:
                    act0 = abs(act0[:, None, :] - act0[:, :, None])
                    act1 = abs(act1[:, None, :] - act1[:, :, None])
                    if edges:
                        act0 = act0[:, *zip(*edges)]
                        act1 = act1[:, *zip(*edges)]
                    else:
                        act0 = act0[:, *np.tril_indices(act0.shape[2], k=-1)]
                        act1 = act1[:, *np.tril_indices(act1.shape[2], k=-1)]
                ERS_ar = get_trial_x_trial(act0, act1,
                                           trial_similarity=trial_similarity)
                n_nans = np.sum(np.isnan(ERS_ar))
                n_non_nans = np.sum(~np.isnan(ERS_ar))

                # plt.imshow(ERS_ar)
                # plt.colorbar()
                # plt.show()

                ERS_same = ERS_ar[np.diag_indices_from(ERS_ar)]
                ERS_same_M = np.nanmean(ERS_same)
                same_Ms.append(ERS_same_M)
                ERS_ar_dif = ERS_ar.copy()
                ERS_ar_dif[np.diag_indices_from(ERS_ar_dif)] = np.nan
                ERS_dif = np.nanmean(ERS_ar_dif, axis=0)
                ERS_dif_M = np.nanmean(ERS_dif)
                dif_Ms.append(ERS_dif_M)
            same_Ms_age.append(same_Ms)
            dif_Ms_age.append(dif_Ms)

        same_Ms_age = np.array(same_Ms_age)
        dif_Ms_age = np.array(dif_Ms_age)
        overall_corr = np.mean([same_Ms_age, dif_Ms_age])

        ef_Ms_age = same_Ms_age - dif_Ms_age
        interaction = ef_Ms_age[:, 0] - ef_Ms_age[:, 1]
        ef_Ms_age = np.concatenate([ef_Ms_age, interaction[:, None]], axis=1)
        ef_gm = np.nanmean(ef_Ms_age, axis=0)
        ef_sd = np.nanstd(ef_Ms_age, axis=0)
        ef_se = ef_sd / np.sqrt(ef_Ms_age.shape[0])
        ef_t = ef_gm / ef_se
        out_str = f'ERS effect ({age}) | ' \
              f'cond0: ef = {ef_gm[0]:.5f} [{ef_se[0]:.5f}], t = {ef_t[0]:.2f} | ' \
              f'cond1: ef = {ef_gm[1]:.5f} [{ef_se[1]:.5f}], t = {ef_t[1]:.2f} | ' \
              f'interaction: t = {ef_t[2]:.2f}'
        print(out_str)

        same_Ms_all.append(same_Ms_age)
        dif_Ms_all.append(dif_Ms_age)
        out_strs.append(out_str)
    return out_strs



def conn_similarity(sn_inc_activity, age2idxs, top_edges_mat,
                    p,
                    variability=True):
    # TODO: maybe control for run effects?
    efs_all = []
    withins_all = []
    betweens_all = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        efs_age = []
        sames_age = []
        difs_age = []
        withins_age = []
        withins_age_multi = []
        betweens_age_multi = []
        betweens_age = []
        bad_sns = {1: [9], 2: []}
        # bad_sns = [9] # nan mismatch within trial
        for sn in tqdm(range(age_sn_inc_act.shape[0])):
            if sn in bad_sns[age]: continue
            efs = []
            sames = []
            difs = []
            smaller_trial = 1e10
            withins = []
            betweens = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = age_sn_inc_act[sn, inc0]
                nans = np.all(np.isnan(act0), axis=0)
                nans_any = np.any(np.isnan(act0), axis=0)
                # if np.sum(~(nans == nans_any)) > 1:
                    # print('NaN mismatch within trial')
                    # act0[:, nans_any] = np.nan
                    # nans = np.all(np.isnan(act0), axis=0)
                    # print(f'bad {age=}, {sn=}')
                n_nans = np.sum(nans)
                n_not_nans = np.sum(~nans)
                # print(f'{age=}, {sn=}, {inc0=}')
                if n_not_nans < 20:
                    print(f'Not enough non-nans: {n_not_nans=}')
                    break

                act0 = act0.T
                act0 = abs(act0[:, None, :] - act0[:, :, None])
                act0 = act0[:, np.tril_indices(act0.shape[2], k=-1)]

                RSM = get_trial_x_trial_RSM(act0, act0, simple_mean=False,
                                            trial_similarity='corr')
                withins.append(np.nanmean(RSM))
                # within_score = []
                # for run0 in range(3):
                #     low0, high0 = run0 * 38, (run0+1) * 38
                #     for run1 in range(3):
                #         low1, high1 = run1 * 38, (run1+1) * 38
                #         count_non_nan = np.sum(~np.isnan(RSM[low0:high0, low1:high1]))
                #         print(f'{run0}, {run1}, {count_non_nan=}')
                #         score = np.nanmean(RSM[low0:high0, low1:high1])
                #         within_score.append(score)
                # withins.append(np.nanmean(within_score))



                # plt.imshow(RSM)
                # plt.show()
                # print(f'{withins[-1]}')
                # quit()
                # p_above = np.mean(np.array(withins_age) > 0)

                for inc1 in range(inc0+1, age_sn_inc_act.shape[1]):
                    if inc0 > inc1:
                        continue
                    act1 = age_sn_inc_act[sn, inc1]
                    nans = np.all(np.isnan(act1), axis=0)


                    act1 = act1.T
                    act1 = abs(act1[:, None, :] - act1[:, :, None])
                    act1 = act1[:, np.tril_indices(act1.shape[2], k=-1)]
                    # act1 = act0

                    RSM = get_trial_x_trial_RSM(act0, act1, simple_mean=False,
                                                trial_similarity='corr')
                    betweens.append(np.nanmean(RSM))

                    # between_score = []
                    # print('between')
                    # for run0 in range(3):
                    #     low0, high0 = run0 * 38, (run0 + 1) * 38
                    #     for run1 in range(3):
                    #         low1, high1 = run1 * 38, (run1 + 1) * 38
                    #         count_non_nan = np.sum(~np.isnan(RSM[low0:high0, low1:high1]))
                    #         print(f'{run0}, {run1}, {count_non_nan=}')
                    #         score = np.nanmean(RSM[low0:high0, low1:high1])
                    #         between_score.append(score)
                    # betweens.append(np.nanmean(between_score))
                    # quit()

                    # print(f'{betweens[-1]=:.3f}')
                    # quit()
            else:
                    # plt.title('between')
                    # plt.imshow(RSM)
                    # plt.show()
                    # act1 = stdize(act1, axis=1, nans=True)
                    # mat01 = act0[:, None, :] * act1[None, :, :]
                    # mat01 = np.nanmean(mat01, axis=2)
                    # between_similarity = np.nanmean(mat01)
                    # betweens.append(between_similarity)
                    # plt.imshow(mat01)
                    # plt.show()
                w = np.mean(withins)
                b = np.mean(betweens)
                withins_age.append(w)
                betweens_age.append(b)
                efs_age.append(b - w)
                withins_age_multi.append(withins)
                betweens_age_multi.append(betweens)

        # print('-')
            # quit()
        withins_age_multi = np.array(withins_age_multi)
        betweens_age_multi = np.array(betweens_age_multi)
        M_withins_age_multi = np.nanmean(withins_age_multi, axis=0)
        M_between_age_multi = np.nanmean(betweens_age_multi, axis=0)

        M_age_same = np.mean(withins_age)
        SE_age_same = np.std(withins_age) / np.sqrt(len(withins_age))
        M_age_dif = np.mean(betweens_age)
        SE_age_dif = np.std(betweens_age) / np.sqrt(len(betweens_age))
        M_age_ef = np.mean(efs_age)
        SD_age_ef = np.std(efs_age)
        SE_age_ef = SD_age_ef / np.sqrt(len(efs_age))
        t_age_ef = M_age_ef / SE_age_ef
        p_age_ef = stats.t.sf(np.abs(t_age_ef), len(efs_age)-1)*2
        # print(f'{withins=}')
        # print(f'{betweens=}')
        # print(f'{withins_age=}')
        # print(f'{betweens_age=}')
        print(f'Subject specific effect ({age}) | '
              f'within: {M_age_same:.3f} [{SE_age_same:.3f}, {M_withins_age_multi}], '
              f'between: {M_age_dif:.3f} [{SE_age_dif:.3f}, {M_between_age_multi}], '
              f't = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        withins_all.append(withins_age)
        betweens_all.append(betweens_age)
        efs_all.append(efs_age)
    t, p = stats.ttest_ind(efs_all[0], efs_all[1])
    print(f'\tYoung vs. Old: t = {t:.3f}, p = {p:.3f}')


def graph_theory(sn_inc_activity, age2idxs, p_top_edges, p, threshold=0.8,
                 measure='omega'):
    print(f'Measure: {measure}')
    age_Ms = []
    age_difs = []
    for age in [1, 2]:
        scores_age = []
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        biggest_of_any = 0
        for sn in range(age_sn_inc_act.shape[0]):
            scores_sn = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = age_sn_inc_act[sn, inc0]
                nan_trials = np.all(np.isnan(act0), axis=0)
                act0 = act0[:, ~nan_trials]
                act0_p = act0[p, :]
                nan_voxels = np.any(np.isnan(act0_p), axis=1)
                act0_p = act0_p[~nan_voxels, :]
                conn0_p = ma.corrcoef(ma.masked_invalid(act0_p))
                conn0_p[np.diag_indices_from(conn0_p)] = np.nan
                thresh = np.nanquantile(conn0_p, threshold)

                conn0_p[conn0_p < thresh] = 0
                conn0_p[conn0_p >= thresh] = 1

                G = nx.from_numpy_array(conn0_p)
                comp = nx.algorithms.components.connected_components(G)
                biggest = set()
                for c in comp:
                    if len(c) > len(biggest):
                        biggest = c

                if len(biggest) < 20 and measure != 'len':
                    # print(f'Bad: {age=}, {sn=}, {inc0=}, {len(biggest)=}')
                    break
                if len(biggest) > biggest_of_any:
                    biggest_of_any = len(biggest)
                    # print(f'Biggest: {age=}, {sn=}, {inc0=}, {len(biggest)=}')
                G = G.subgraph(biggest)
                # print('Calculating small worldness')
                # smol = nx.sigma(G, niter=10, nrand=2)
                if measure == 'len':
                    score = len(biggest)
                elif measure == 'shortest':
                    score = nx.average_shortest_path_length(G)
                    if len(biggest) < biggest_of_any: # TODO: better
                        score_d = nx.shortest_path_length(G)
                        worst_M_score = 0
                        for node, d in score_d:
                            node_score = np.mean(list(d.values()))
                            if node_score > worst_M_score:
                                worst_M_score = node_score
                        if len(biggest) < biggest_of_any:
                            missing = biggest_of_any - len(biggest)
                            score_new = (worst_M_score * missing +
                                         score * len(biggest)) / biggest_of_any
                            # print(f'{score} | {score_new}')
                            score = score_new
                elif measure == 'clustering':
                    score = nx.average_clustering(G)
                elif measure == 'closeness_centrality':
                    score = nx.closeness_centrality(G)
                    print(f'Closeness: {age=}, {sn=}, {inc0=} | {score=:.3f}')
                elif measure == 'sigma':
                    score = nx.sigma(G, niter=10, nrand=5, seed=0)
                    print(f'Sigma: {age=}, {sn=}, {inc0=} | {score=:.3f}')
                elif measure == 'omega':
                    score = nx.omega(G, niter=10, nrand=5, seed=0)
                    print(f'Omega: {age=}, {sn=}, {inc0=} | {score=:.3f}')

                else:
                    raise NotImplementedError
                scores_sn.append(score)
            else:
                scores_age.append(scores_sn)
        scores_age = np.array(scores_age)
        # print(f'{scores_age.shape=}')
        # print(f'{scores_age=}')
        age_M = np.nanmean(scores_age, axis=1)
        age_Ms.append(age_M)

        Ms = np.nanmean(scores_age, axis=0)
        SEs = np.nanstd(scores_age, axis=0) / np.sqrt(scores_age.shape[0])
        desc_str = ''
        for cond in range(scores_age.shape[1]):
            desc_str += f'{Ms[cond]:.3f} [{SEs[cond]:.3f}], '
        desc_str = desc_str[:-2]
        print(f'{age=}: {desc_str}')

        comparisons = itertools.combinations(range(scores_age.shape[1]), 2)
        comparisons = list(comparisons)
        difs_all = []
        for c in comparisons:
            difs = scores_age[:, c[0]] - scores_age[:, c[1]]
            difs_all.append(difs)
            SE_difs = np.nanstd(difs) / np.sqrt(difs.shape[0])
            t = np.nanmean(difs) / SE_difs
            p_val = stats.t.sf(np.abs(t), difs.shape[0] - 1) * 2
            print(f'\t{c[0]} vs. {c[1]}: t = {t:.3f}, p = {p_val:.3f}')
        age_difs.append(difs_all)
    M_young = np.nanmean(age_Ms[0])
    M_old = np.nanmean(age_Ms[1])
    t, p_val = stats.ttest_ind(age_Ms[0], age_Ms[1])
    print(f'Young ({M_young:.3f}) vs. Old ({M_old:.3f}): '
          f't = {t:.3f}, p = {p_val:.3f}')
    # print(len(difs_all))
    # print(len(difs_all[0]))
    # print(len(difs_all[0][0]))
    # quit()
    for c_i in range(len(age_difs[0])):
        difs_young = age_difs[0][c_i]
        difs_old = age_difs[1][c_i]
        t, p_val = stats.ttest_ind(difs_young, difs_old)
        print(f'\tAge x Dif ({c_i}): t = {t:.3f}, p = {p_val:.3f}')
        difs_both = np.concatenate([difs_young, difs_old])
        M_both = np.nanmean(difs_both)
        SE_both = np.nanstd(difs_both) / np.sqrt(difs_both.shape[0])
        t_both = M_both / SE_both
        print(f'\t\tBoth ({c_i}): {M_both:.3f} [{SE_both:.3f}], '
              f't = {t_both:.3f}')
