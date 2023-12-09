import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from tqdm import tqdm

from conn_RSA import get_trial_x_trial_RSM
from fMRI_proc import within_run_to_nan
from utils import stdize
import scipy.stats as stats

def corrcoef_na(A, B):
    return ma.corrcoef(ma.masked_invalid(A), ma.masked_invalid(B))

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
                # print(nans)
                # print(nans == nans_any)
                # print('inc0:', np.argwhere(~nans).T)
                # print(act0)

                # plt.imshow(act0)
                # plt.show()
                # quit()
                # act0 = act0[:, ~nans]
                act0 = act0.T
                # M_sans = np.nanmean(act0, axis=0)
                # print(act0.shape)
                # M_sans = M_sans[:, None]

                # print(M_sans.shape)
                # quit()
                # print(act0.shape)
                # quit()
                # EX = np.nanmean(act0, axis=0)
                # EX0_EX1 = EX[:, None] * EX[None, :]
                # act0 = act0[:, None, :] * act0[:, :, None] - EX0_EX1[None, :, :]
                # act0 = stdize(act0, axis=0, nans=True)
                # act0 = abs(act0[:, None, :] - act0[:, :, None])
                # act0 = act0[:, *np.tril_indices(act0.shape[2], k=1)]
                # mat00 = ma.corrcoef(ma.masked_invalid(act0))
                # mat00 = within_run_to_nan(mat00)
                # mat00[np.diag_indices_from(mat00)] = np.nan
                # RSM = mat00
                RSM = get_trial_x_trial_RSM(act0, simple_mean=False,
                                            trial_similarity='corr')
                withins.append(np.nanmean(RSM))
                p_above = np.mean(np.array(withins_age) > 0)
                # print(f'{withins=}')
                # print(f'{p_above=:.3f}')
                # if np.nanmean(RSM) < -100:
                #     plt.imshow(RSM)
                #     plt.colorbar()
                #     plt.title(f'{np.nanmean(RSM)=}')
                #     plt.show()
                #     quit()
                # continue
                # plt.title('within')
                # plt.imshow(RSM)
                # plt.show()
                # print(act0.shape)
                # quit()
                # act0 = abs(act0[:, None, :] * act0[:, :, None])
                # act0 = stdize(act0, axis=1, nans=True)
                # mat00 = ma.corrcoef(ma.masked_invalid(act0))
                # mat00[np.diag_indices_from(mat00)] = np.nan
                # within_similarity = np.nanmean(mat00)
                # withins.append(within_similarity)

                for inc1 in range(inc0+1, age_sn_inc_act.shape[1]):
                    if inc0 > inc1:
                        continue
                    act1 = age_sn_inc_act[sn, inc1]
                    nans = np.all(np.isnan(act1), axis=0)
                    # nans_any = np.any(np.isnan(act1), axis=0)
                    # if np.sum(~(nans == nans_any)) > 1:
                    #     print('NaN mismatch within trial')
                    #     act1[:, nans_any] = np.nan
                    #     nans = np.all(np.isnan(act1), axis=0)
                    # print(nans)
                    # print(nans == nans_any)
                    # quit()
                    # print('inc1:', np.argwhere(~nans).T)
                    # print(act1)
                    # act1 = act1[:, ~nans]
                    act1 = act1.T
                    # print(act1.shape)
                    # quit()

                    # EX = np.nanmean(act1, axis=0)
                    # EX0_EX1 = EX[:, None] * EX[None, :]
                    # act1 = act1[:, None, :] * act1[:, :, None] - EX0_EX1[None, :, :]

                    # act1 = stdize(act1, axis=0, nans=True)
                    # act1 = abs(act1[:, None, :] * act1[:, :, None])
                    # act1 = abs(act1[:, None, :] - act1[:, :, None])
                    # act1 = act1[:, *np.tril_indices(act1.shape[2], k=1)]
                    RSM = get_trial_x_trial_RSM(act0, act1,
                                                trial_similarity='corr')
                    betweens.append(np.nanmean(RSM))
                    # quit()

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
            # quit()

        M_age_same = np.mean(withins_age)
        M_age_dif = np.mean(betweens_age)
        M_age_ef = np.mean(efs_age)
        SD_age_ef = np.std(efs_age)
        SE_age_ef = SD_age_ef / np.sqrt(len(efs_age))
        t_age_ef = M_age_ef / SE_age_ef
        p_age_ef = stats.t.sf(np.abs(t_age_ef), len(efs_age)-1)*2
        # print(f'{withins=}')
        # print(f'{betweens=}')
        print(f'{withins_age=}')
        print(f'{betweens_age=}')
        print(f'Subject specific effect ({age}) | within: {M_age_same:.3f}, '
              f'between: {M_age_dif:.3f}, t = {t_age_ef:.3f}, p = {p_age_ef:.3f}')
        withins_all.append(withins_age)
        betweens_all.append(betweens_age)
        efs_all.append(efs_age)
    t, p = stats.ttest_ind(efs_all[0], efs_all[1])
    print(f'\tYoung vs. Old: t = {t:.3f}, p = {p:.3f}')



if __name__ == '__main__':
    pass






