import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from connsearch import RepeatedStratifiedGroupKFold
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

from conn_RSA import get_trial_x_trial_RSM
from fMRI_proc import within_run_to_nan
from utils import stdize
import scipy.stats as stats

def corrcoef_na(A, B):
    return ma.corrcoef(ma.masked_invalid(A), ma.masked_invalid(B))

def conn_classifier(sn_inc_activity, age2idxs, top_edges_mat,
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
        GM_acc = []
        for sn in tqdm(range(age_sn_inc_act.shape[0])):
            if sn in bad_sns[age]: continue
            efs = []
            sames = []
            difs = []
            smaller_trial = 1e10
            withins = []
            betweens = []
            assert age_sn_inc_act.shape[1] == 2
            X = []
            Y = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = age_sn_inc_act[sn, inc0]
                nan_trials = np.all(np.isnan(act0), axis=0)
                act0 = act0[:, ~nan_trials]
                nan_voxels = np.any(np.isnan(act0), axis=1)
                act0 = act0[~nan_voxels, :]
                act0 = act0.T
                # print(act0.shape)
                # quit()
                # act0 = stdize(act0, axis=1)
                # act0 = abs(act0[:, None, :] - act0[:, :, None])
                # act0 = act0[:, None, :] * act0[:, :, None]
                # act0 = act0[:, *np.tril_indices(act0.shape[2], k=1)]
                # act0 = act0[~nan_trials, :]
                # print(act0)
                X.append(act0)
                # plt.imshow(act0)
                # plt.show()
                Y.extend([inc0] * act0.shape[0])
            X = np.concatenate(X, axis=0)
            # nan_cols = np.any(np.isnan(X), axis=1)
            # plt.imshow(X)
            # plt.show()

            Y = np.array(Y)
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=100)
            clf = SVC(kernel='linear')
            acc = cross_val_score(clf, X, Y, cv=cv)
            acc = np.mean(acc)
            # print(f'{acc=:.3f}')
            GM_acc.append(acc)
            grand_mean = np.mean(GM_acc)
            grand_SE = np.std(GM_acc) / np.sqrt(len(GM_acc))
            grand_t = (grand_mean - 0.5) / grand_SE
            print(f'age {age}: {grand_mean=:.3f} [{grand_SE=:.3f}], {grand_t=:.3f}')

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



if __name__ == '__main__':
    pass






