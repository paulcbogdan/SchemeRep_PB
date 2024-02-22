from pathlib import Path

import numpy as np
from connsearch.report import plot_ROI_scores
from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedGroupKFold
from sklearn.svm import SVC

from atlas_utils import get_atlas
from old.networks import load_FC_for_Lifu
from old.classifiers import stratify
from utils import pickle_wrap
from tqdm import tqdm
from collections import defaultdict
from connsearch import print_list_stats
import scipy.stats as stats

def shuffle_Y(Y, groups):
    for grp in np.unique(groups):
        idxs = groups == grp
        Y[idxs] = np.random.permutation(Y[idxs])
    return Y

def permutation_test(n_voxels=10, linear=False, full_perms=100):
    kwargs = {'fp': 'obj4_fMRI',
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'odd_even': False,
              'voxelwise': True
              }
    ROI2act, Y_all, groups_all, age2idxs = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../cache')
    Y_all = np.array(Y_all)
    groups_all = np.array(groups_all)
    age2accs = defaultdict(list)
    for _ in range(full_perms):
        for age in [1, 2]:
            for ROI, X_all in ROI2act.items():
                X, Y, groups = preproc_XYg(age2idxs, age, Y_all, groups_all,
                                           X_all)
                if X.shape[1] == 0:
                    print(f'No voxels for {ROI} ({age=})')
                    continue
                elif X.shape[1] < 10:
                    print(f'Few voxels for {ROI} ({age=})')
                if X.shape[1] < n_voxels:
                    continue
                rand_voxels = np.random.choice(X.shape[1], n_voxels,
                                               replace=False)
                X = X[:, rand_voxels]
                acc = RepeatedGroupKfold(X, Y, groups, linear, repeats=100)
                age2accs[age].append(acc)
                print(f'{age=}')
                print_list_stats(age2accs[age])
    # 10 repeats
    #       age=1
    #   p(under 50%): 0.454 | p(above 50%): 0.546
    #   Percentile: Accuracy | 1.0: 0.4715, 0.75: 0.4953, 0.5: 0.5012,
    #   0.25: 0.5074, 0.1: 0.5144, 0.05: 0.5194,
    #   0.01: 0.5270, 0.005: 0.5277, 0.001: 0.5432
    #       age=2
    #   p(under 50%): 0.412 | p(above 50%): 0.586
    #   Percentile: Accuracy | 1.0: 0.4790, 0.75: 0.4963, 0.5: 0.5019,
    #   0.25: 0.5079, 0.1: 0.5132, 0.05: 0.5178,
    #   0.01: 0.5278, 0.005: 0.5304, 0.001: 0.5321

def preproc_XYg(age2idxs, age, Y_all, groups_all, X_all):
    age_idxs = age2idxs[age]
    age_idxs_ext = []
    for v in age_idxs:
        for i in range(114):
            age_idxs_ext.append(v * 114 + i)
    Y = Y_all[age_idxs_ext]
    groups = groups_all[age_idxs_ext]
    X = X_all[age_idxs, :]
    X, Y, groups = prune_nans(X, Y, groups)
    X = interp(X)
    return X, Y, groups

def prune_nans(X, Y, groups):
    nan_trials = np.isnan(Y)
    Y = Y[~nan_trials]
    groups = groups[~nan_trials]
    X = X.reshape(-1, X.shape[-1])
    X = X[~nan_trials, :]
    return X, Y, groups

def interp(X, thresh=.3):
    p_nan_voxel = np.sum(np.isnan(X), axis=0) / X.shape[0]
    X = X[:, p_nan_voxel < thresh]
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)  # remove outliers?
    return X

def RepeatedGroupKfold(X, Y, groups, linear, repeats=10):
    accs = []
    for _ in tqdm(range(repeats), desc='Repeating kfold', position=0,
                  leave=True):
        grps_unq = np.sort(np.unique(groups))
        grps_unq_ = np.sort(np.unique(groups))
        np.random.shuffle(grps_unq_)
        grp_mapper = {}
        for i, grp in enumerate(grps_unq):
            grp_mapper[grp] = grps_unq_[i]
        groups = np.array([grp_mapper[grp] for grp in groups])
        if len(np.unique(groups)) % 2 != 0:
            X_ = X[groups != 0]
            Y_ = Y[groups != 0]
            groups_ = groups[groups != 0]
        else:
            X_ = X
            Y_ = Y
            groups_ = groups
        cv = GroupKFold(n_splits=2)
        clf = SVC(kernel='linear' if linear else 'rbf')
        acc = cross_val_score(clf, X_, Y_, cv=cv, groups=groups_)
        acc = np.mean(acc)
        accs.append(acc)
        # if len(accs) > 2 and np.mean(accs) < .53:
        #     break
    acc = np.mean(accs)
    return acc

def voxelwise_vis_clf(linear=True, group_level=True, n_repeats=25):
    kwargs = {'fp': 'obj4_fMRI',
              'split': False,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 3),
              'odd_even': False,
              # 'regionwise': True
              'voxelwise': True
              }
    ROI2act, Y_all, groups_all, age2idxs = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='../cache')
    Y_all = np.array(Y_all)
    groups_all = np.array(groups_all)
    # print(groups)
    # quit()


    # ROI2act['vis'] = []
    # vis_ROIs = [80, 81, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 104, 105, 106, 107, 134, 135, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
    # for i, (ROI, X_all) in enumerate(ROI2act.items()):
    #     if i in vis_ROIs:
    #         ROI2act['vis'].append(np.nanmean(X_all, axis=-1))
    #         print(f'{ROI=}')
    # # ROI2act['vis'] = np.concatenate(ROI2act['vis'], axis=2)
    # ROI2act['vis'] = np.array(ROI2act['vis']).transpose((1, 2, 0))
    # print(ROI2act['vis'].shape)
    # quit()
    # vis_ROIs = ['LOC', 'EVC', 'sOcG']
    # for ROI, X_vall in ROI2act.items():
    #     if ROI in vis_ROIs:
    #         ROI2act['vis'].append(X_vall)
    # ROI2act['vis'] = np.concatenate(ROI2act['vis'], axis=2)

    atlas = get_atlas(schaefer=kwargs['atlas_name'] == 'schaefer')
    regions = atlas['tick_labels']
    # region2act = {}
    # region2n = {}
    # for region in regions:
    #     region2act[region] = []
    #     for ROI, X_all in ROI2act.items():
    #         if region in ROI:
    #             region2act[region].append(X_all)
    #     region2n[region] = len(region2act[region])
    #     region2act[region] = np.concatenate(region2act[region], axis=2)
    # ROI2act = region2act

    age2accs = {1: [], 2: []}
    for ROI, X_all in ROI2act.items():
        # print(f'{ROI}, {X_all.shape=}')
        # quit()
        # if 'IPL' not in ROI:
        #     continue
        # if ROI != 'vis':
        #     continue
        age_scores = {}
        for age in [1, 2]:
            if group_level:
                X, Y, groups = preproc_XYg(age2idxs, age, Y_all, groups_all, X_all)
                if X.shape[1] == 0:
                    print(f'No voxels for {ROI} ({age=})')
                    continue
                elif X.shape[1] < 10:
                    print(f'Few voxels for {ROI} ({age=})')
                score = RepeatedGroupKfold(X, Y, groups, linear,
                                         repeats=n_repeats)
            else:
                age_idxs = age2idxs[age]
                age_idxs_ext = []
                for v in age_idxs:
                    for i in range(114):
                        age_idxs_ext.append(v * 114 + i)
                Y = Y_all[age_idxs_ext]
                X = X_all[age_idxs, :]

                # print(X.shape)
                # quit()
                Y = Y.reshape(X.shape[0], 114)
                accs = []
                for sn in range(X.shape[0]):

                    x = X[sn, :, :]
                    acc = np.nanmean(x)
                    # continue
                    # y = Y[sn, :]
                    # acc = np.nanmean(x[y == 0]) - np.nanmean(x[y == 1])
                    accs.append(acc)
                    continue
                    grp = [0] * 38 + [1] * 38 + [2] * 38
                    # grp = [0] * 18 + [1] * 18 + [2] * 18 + \
                    #       [3] * 18 + [4] * 18 + [5] * 18 + [6] * 6
                    grp = np.array(grp)
                    # print(y)
                    x, y, grp = prune_nans(x, y, grp)
                    accs_sn = []
                    for _ in range(10):
                        grp_ = np.random.permutation(grp)
                        x_, y_, grp_ = stratify(x, y, grp_)
                        # x_, y_, grp_ = x, y, grp
                        x_ = np.array(x_)
                        x_ = interp(x_)
                        if x_.shape[1] == 0:
                            print(f'No voxels for {ROI} ({age=})')
                            continue
                        # len(np.unique(grp))
                        cv = StratifiedGroupKFold(n_splits=len(np.unique(grp)))
                        clf = SVC(kernel='linear' if linear else 'rbf')
                        acc = cross_val_score(clf, x_, y_, cv=cv, groups=grp_)
                        acc = np.mean(acc)
                        acc -= 0.5
                        accs_sn.append(acc)
                    accs.append(np.mean(accs_sn))
                accs = np.array(accs)
                age_scores[age] = accs
                acc_m = np.nanmean(accs)
                acc_sd = np.nanstd(accs)
                acc_n = np.sum(~np.isnan(accs))
                acc_se = acc_sd / np.sqrt(acc_n)
                score = (acc_m) / acc_se

            age2accs[age].append(score)
            # age2accs[age].append([score]*region2n[ROI])

            print(f'{ROI} ({age=}): {score=:.3f}')
        dif_t, p = stats.ttest_ind(age_scores[1], age_scores[2])
        print(f'{dif_t=}')
        print('-')
    # quit()
    age2str = {0: 'YA & OA', 1: 'YA', 2: 'OA'}
    accs_both = np.array(age2accs[1] + age2accs[2])
    if group_level:
        vmin = np.nanquantile(accs_both, .01)
        vmin = .51
        vmax = np.nanquantile(accs_both, .975)
    else:
        # vmin = 2.0
        vmin = np.nanquantile(accs_both, .025)
        vmax = np.nanquantile(accs_both, .975)

    for age in [1, 2]:
        fp_pic = f'clf_activity_{age2str[age]}.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        plot_ROI_scores(age2accs[age], atlas['coords'], fp_out=fp_pic,
                        show=True, title=f'activation (I & C) | '
                                         f'{age2str[age]}',
                        vmin=vmin, vmax=vmax, cmap='viridis')

if __name__ == '__main__':
    voxelwise_vis_clf(group_level=False)
    # permutation_test()


