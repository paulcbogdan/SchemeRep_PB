import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold, \
    GroupKFold
from sklearn.svm import SVC
from tqdm import tqdm

from utils import stdize


def corrcoef_na(A, B):
    return ma.corrcoef(ma.masked_invalid(A), ma.masked_invalid(B))

def stratify(X, Y, groups=None):
    labels = np.unique(Y)
    X_new = []
    Y_new = []
    n_pre = len(X)
    if groups is not None:
        groups_new = []
        for group in np.unique(groups):
            Y_grp = Y[groups == group]
            fewest_labels = np.min([np.sum(Y_grp == label) for label in labels])
            for label in labels:
                X_grp_label = X[groups == group][Y_grp == label][:fewest_labels]
                X_new.extend(X_grp_label)
                Y_new.extend(Y[groups == group][Y_grp == label][:fewest_labels])
                groups_new.extend([group]*len(X_grp_label))
        # n_post = len(X_new)
        # print(f'Pre-stratification: {n_pre} samples. '
        #       f'Post-stratification: {n_post} samples.')
        return X_new, Y_new, groups_new
    else:
        fewest_labels = np.min([np.sum(Y == label) for label in labels])
        for label in labels:
            X_label = X[Y == label][:fewest_labels]
            X_new.extend(X_label)
            Y_new.extend(Y[Y == label][:fewest_labels])
        n_post = len(X_new)
        # print(f'Number dropped to stratify: '
        #       f'{n_pre} - {n_post} = {n_pre - n_post}')
        return X_new, Y_new

def edges_classifier(sn_inc_activity, age2idxs, edges):
    # for age in [1, 2]:
    #     sn_idxs = age2idxs[age]
    #     age_sn_inc_act = sn_inc_conn[sn_idxs]
    #     GM_acc = []
    #     for sn in tqdm(range(sn_inc_conn.shape[0])):
    #         # if sn in bad_sns[age]: continue
    #         assert sn_inc_conn.shape[1] in [2, 3]
    #         sn_conn = sn_inc_conn[sn]
    #         for inc0 in range(sn_conn.shape[1]):
    #             conn0 = sn_conn[inc0]
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        GM_acc = []
        for sn in tqdm(range(age_sn_inc_act.shape[0])):
            assert age_sn_inc_act.shape[1] in [2, 3]
            X = []
            Y = []
            sn_act = age_sn_inc_act[sn]
            groups = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = sn_act[inc0]

def partition_group_clf(sn_inc_activity, age2idxs, edges,
                         stratification_strategy=1,
                         trial_mapper=None, split=1,
                        acttivity=False, linear=True):
    empty_mapper = len(trial_mapper) == 0
    # TODO: Double check that the congruent/incongruent labels are correct?
    ts = []
    bad_sns = {1: [3, 4, 6, 21, 27],
               2: [4, 5, 15, 16, 18, 24]}
    if acttivity:
        edges = list(set(x for sl in edges for x in sl))
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        GM_acc = []
        groups = []
        X = []
        Y = []

        for sn in range(age_sn_inc_act.shape[0]):
            # if sn in bad_sns[age]: continue
            assert age_sn_inc_act.shape[1] in [2, 3]
            sn_act = age_sn_inc_act[sn]
            bad = False
            Y_sn = []
            X_sn = []
            groups_sn = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = sn_act[inc0]
                act0 = act0.T
                num_runs = act0.shape[0] // 38
                for run in range(num_runs):
                    if empty_mapper:
                        run_low = run * 38
                        run_high = (run + 1) * 38
                        trial_set = act0[run_low:run_high]
                        non_nan_bool = ~np.isnan(trial_set[:, 0])
                        non_nan_trials = np.argwhere(non_nan_bool).squeeze()
                        for spl in range(split):
                            try:
                                non_nan_trials_spl = non_nan_trials[
                                spl::split]
                            except IndexError:
                                bad = True
                                break

                            # quit()
                            trial_set_ = trial_set[non_nan_trials_spl]
                            if acttivity:
                                trial_set_ = np.nanmean(trial_set_, axis=0)
                                trial_mapper[(age, sn, inc0, run, spl)] = trial_set_
                            else:
                                trial_set_ = stdize(trial_set_, axis=0, nans=True)
                                trial_set_ = trial_set_[:, None, :] * trial_set_[:, :, None]
                                trial_set_ = np.nanmean(trial_set_, axis=0)
                                trial_mapper[(age, sn, inc0, run, spl)] = trial_set_
                    if bad:
                        break
                    trial_set = []
                    for spl in range(split):
                        trial_set.append(trial_mapper[(age, sn, inc0, run, spl)])
                    trial_set = np.array(trial_set)
                    # print(trial_set.shape)
                    if edges:
                        if acttivity:
                            trial_set = trial_set[:, edges]
                        else:
                            edges_ = zip(*edges)
                            trial_set = trial_set[:, *edges_]
                    else:
                        trial_set = trial_set[:, *np.tril_indices(
                            trial_set.shape[2], k=-1)]
                    # print(trial_set.shape)
                    # quit()

                    # trial_set = trial_set[:10]
                    # nan_trials = np.all(np.isnan(trial_set), axis=1)
                    # trial_set = trial_set[~nan_trials]
                    # trial_set = list(trial_set)
                    # trial_sets.extend(trial_set)
                    n_nans = np.sum(np.isnan(trial_set))
                    if n_nans > 200:
                        # TODO: see if there's a way to avoid so many drops
                        print(f'Bad! {age} : {sn}, {run}, {inc0} = {n_nans}')
                        bad = True
                    trial_set = list(trial_set)
                    X_sn.extend(trial_set)
                    # X_sn.append(trial_set)
                    groups_sn.extend([sn] * len(trial_set))
                    # groups_sn.append(sn)
                    Y_sn.extend([inc0] * len(trial_set))
                    # Y_sn.append(inc0)
            if not bad:
                X.extend(X_sn)
                Y.extend(Y_sn)
                groups.extend(groups_sn)
        X = np.array(X)
        Y = np.array(Y)

        nan_edges = np.any(np.isnan(X), axis=0)

        n_nan_edges = np.sum(nan_edges)

        if len(X.shape) < 2 or X.shape[1] < 1:  # n_nan_edges > X.shape[1] // 2:
            print(f'Fail too many nans ({n_nan_edges}): {age=}, {sn=} | '
                  f'{X.shape=}')
            return [np.nan, np.nan]

        X = X[:, ~nan_edges]
        # X = X[:, :2]
        X = np.random.normal(size=X.shape)
        print(f'{X.shape=}')


        Y = np.array(Y)
        groups = np.array(groups)
        # print(f'{Y=}')
        # print(f'{len(Y)=}')
        # print(f'{groups=}')
        if stratification_strategy == 1:
            accs = []
            for repeat in range(25):
                grps_unq = np.sort(np.unique(groups))
                grps_unq_ = np.sort(np.unique(groups))
                np.random.shuffle(grps_unq_)
                grp_mapper = {}
                for i, grp in enumerate(grps_unq):
                    grp_mapper[grp] = grps_unq_[i]
                groups = np.array([grp_mapper[grp] for grp in groups])
                if len(np.unique(groups)) % 2 != 0:
                    X = X[groups != 0]
                    Y = Y[groups != 0]
                    groups = groups[groups != 0]
                # idxs = np.arange(len(Y))
                # np.random.shuffle(idxs)
                # X = X[idxs]
                # Y = Y[idxs]
                # groups = groups[idxs]
                num_groups = len(np.unique(groups))
                # cv = StratifiedGroupKFold(n_splits=2)
                cv = GroupKFold(n_splits=2)
                clf = SVC(kernel='linear' if linear else 'rbf')
                acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
                acc = np.mean(acc)
                accs.append(acc)
            print(f'{accs=}')

            acc = np.mean(accs)
        elif stratification_strategy in [2, 3]:
            num_groups = len(np.unique(groups))
            for g in range(num_groups):
                X[groups == g] -= np.mean(X[groups == g], axis=0)
            # print(f'{len(Y)=}')

            repeats = 10 if stratification_strategy == 2 else 50
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=repeats)
            clf = SVC(kernel='linear' if linear else 'rbf')
            acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
            acc = np.mean(acc)

        GM_acc.append(acc)
        print(f'age {age}: {acc=:.3f}')
        ts.append(acc)

    return ts

def partition_classifier(sn_inc_activity, age2idxs, edges,
                         stratification_strategy=2,
                         trial_mapper=None, activity=False,
                         linear=True):
    # print(sn_inc_activity.shape)
    # quit()
    # sn_inc_activity = sn_inc_activity[:, :, :, :114]
    # sn_inc_activity = sn_inc_activity[:, :, :, ::2]

    # If you group by run, the data must be sorted properly
    if activity:
        edges = list(set(x for sl in edges for x in sl))
    empty_mapper = len(trial_mapper) == 0
    # TODO: Double check that the congruent/incongruent labels are correct?
    ts = []
    for age in [1, 2]:
        sn_idxs = age2idxs[age]
        age_sn_inc_act = sn_inc_activity[sn_idxs]
        GM_acc = []
        for sn in range(age_sn_inc_act.shape[0]):
            # if sn in bad_sns[age]: continue
            assert age_sn_inc_act.shape[1] in [2, 3]
            X = []
            Y = []
            sn_act = age_sn_inc_act[sn]
            groups = []
            for inc0 in range(age_sn_inc_act.shape[1]):
                act0 = sn_act[inc0]

                # nan_trials = np.all(np.isnan(act0), axis=0)
                # nan_voxels = np.any(np.isnan(act0[:, ~nan_trials]), axis=1)
                # act0 = act0[~nan_voxels, :]
                act0 = act0.T

                trial_sets = []
                num_runs = act0.shape[0] // 38
                # print(f'{num_runs=}')
                # print(f'{act0.shape=}')
                # quit()

                def get_trial_set_multi_perm(trial_set_clean, n_perm_=10,
                                             n_length=7):
                    trials_idxs = np.arange(trial_set_clean.shape[0])
                    trials_idxs_ = trials_idxs.copy()
                    exs = []
                    if n_length == 7:
                        if len(trials_idxs_) < 4:
                            print(f'Tiny tiny trial_idxs_: {len(trials_idxs_)}')
                            return []
                        elif len(trials_idxs_) < 5:
                            return get_trial_set_multi_perm(trial_set_clean, n_perm_=10,
                                                 n_length=3)
                        elif len(trials_idxs_) < 6:
                            return get_trial_set_multi_perm(trial_set_clean, n_perm_=10,
                                                 n_length=4)
                        elif len(trials_idxs_) < 7:
                            return get_trial_set_multi_perm(trial_set_clean, n_perm_=10,
                                                 n_length=5)
                        elif len(trials_idxs_) < 8:
                            return get_trial_set_multi_perm(trial_set_clean, n_perm_=10,
                                                 n_length=6)

                    while len(exs) < n_perm_:
                        idxs = np.random.choice(trials_idxs_, size=n_length,
                                               replace=False)
                        trials_idxs_ = np.setdiff1d(trials_idxs_, idxs)
                        if len(trials_idxs_) < n_length:
                            trials_idxs_ = trials_idxs.copy()
                        trial_set_ex = trial_set_clean[idxs]
                        trial_set_ex = stdize(trial_set_ex, axis=0)
                        trial_set_ex = trial_set_ex[:, None, :] * \
                                       trial_set_ex[:, :, None]
                        trial_set_ex = np.nanmean(trial_set_ex, axis=0)
                        exs.append(trial_set_ex)
                    return np.array(exs)



                for run in range(num_runs):
                    if empty_mapper:
                        run_low = run * 38
                        run_high = (run + 1) * 38
                        trial_set = act0[run_low:run_high]
                        if activity:
                            pass
                        else:
                            # trial_set_lower = sn_act[:, :, :run_low]
                            # trial_set_upper = sn_act[:, :, run_high:]
                            # trial_set_else = np.concatenate(
                            #     [trial_set_lower, trial_set_upper], axis=2)
                            # # trial_set_else = sn_act
                            # M_else = np.nanmean(np.nanmean(trial_set_else, axis=2),
                            #                     axis=0)
                            # # TODO: improve SD_else to be a weighted std, such that
                            # #   0 and 1 are both weighted equally (not by # trials)
                            # SD_else = np.nanstd(trial_set_else, axis=(0, 2))
                            # # trial_set = trial_set - M_else
                            # # trial_set /= SD_else
                            #
                            # trial_set = abs(trial_set[:, None, :] -
                            #                 trial_set[:, :, None])
                            non_nan_bool = ~np.isnan(trial_set[:, 0])
                            # non_nan_trials = np.argwhere(non_nan_bool).squeeze()
                            # print(trial_set.shape)
                            # quit()
                            # try:
                            trial_set = get_trial_set_multi_perm(
                                trial_set[non_nan_bool, :])
                            # except ValueError:
                            #     trial_set = []
                            #     continue

                        trial_mapper[(age, sn, inc0, run)] = trial_set
                    else:
                        trial_set = trial_mapper[(age, sn, inc0, run)]
                    if len(trial_set) == 0:
                        continue
                    if edges:
                        if activity:
                            trial_set = trial_set[:, edges]
                        else:
                            edges_ = zip(*edges)
                            edges_ = list(edges_)
                            trial_set = trial_set[:, *edges_]
                    else:
                        trial_set = trial_set[:, *np.tril_indices(
                            trial_set.shape[2], k=-1)]
                    nan_trials = np.all(np.isnan(trial_set), axis=1)
                    trial_set = trial_set[~nan_trials]
                    # print(f'{num_nans=}')

                    trial_set = list(trial_set)
                    trial_sets.extend(trial_set)
                    # print(f'{inc0} | {run=}: {len(trial_set)}')
                    # groups.append(run)
                    groups.extend([run] * len(trial_set))
                    # run_groups = [run] * (len(trial_set) // 2) + \
                    #              [run + 3] * (len(trial_set) - len(trial_set) // 2)
                    # groups.extend(run_groups)
                act0 = np.array(trial_sets)
                X.append(act0)
                # print(f'{act0.shape=}')
                Y.extend([inc0] * act0.shape[0])


            # quit()
            try:
                X = np.concatenate(X, axis=0)
            except ValueError:
                print(f'Fail not enough data: {age=}, {sn=}')
                continue

            nan_edges = np.any(np.isnan(X), axis=0)
            n_nan_edges = np.sum(nan_edges)

            if len(X.shape) < 2 or X.shape[1] < 5:#n_nan_edges > X.shape[1] // 2:
                print(f'Fail too many nans ({n_nan_edges}): {age=}, {sn=} | '
                      f'{X.shape=}')
                return [np.nan, np.nan]

            X = X[:, ~nan_edges]
            # print(f'{X=}')
            # print(f'individual: {X.shape=}')
            # print(f'{np.sum(nan_edges)=}')
            # print('TEST')

            Y = np.array(Y)
            # print(f'{Y=}')
            # print(f'{len(Y)=}')
            # print(f'{groups=}')
            if stratification_strategy == 1:
                X, Y, groups = stratify(X, Y, groups=groups)
                if len(np.unique(groups)) % 3 != 0:
                    print(f'Not enough data stratify ({age=}, {sn=})')
                    continue
                # num_groups = len(np.unique(groups))
                # for g in range(num_groups):
                #     X[groups == g] -= np.mean(X[groups == g], axis=0)
                # groups = [grp % 3 for grp in groups]

                cv = StratifiedGroupKFold(n_splits=len(np.unique(groups)))
                clf = SVC(kernel='linear' if linear else 'rbf')
                acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
                acc = np.mean(acc)
            elif stratification_strategy in [2, 3]:
                num_groups = len(np.unique(groups))
                # TODO: need the stratification for the mean subtraction
                #    but this isn't optimal as it throws out some data
                X, Y, groups = stratify(X, Y, groups=groups)

                for g in range(num_groups):
                    X[groups == g] -= np.mean(X[groups == g], axis=0)

                repeats = 10 if stratification_strategy == 2 else 50
                cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=repeats)
                clf = SVC(kernel='linear' if linear else 'rbf')
                acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
                acc = np.mean(acc)
            elif stratification_strategy == 4:
                # run-by-run classification...
                accs_groups = []
                for group in np.unique(groups):
                    # print(X.shape)
                    X_grp = X[groups == group]
                    # print(X_grp.shape)
                    Y_grp = Y[groups == group]
                    X_grp, Y_grp = stratify(X_grp, Y_grp)

                    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10)
                    clf = SVC(kernel='linear' if linear else 'rbf')
                    acc = cross_val_score(clf, X_grp, Y_grp, cv=cv,
                                         )
                    acc = np.mean(acc)
                    accs_groups.append(acc)
                acc = np.mean(accs_groups)

            GM_acc.append(acc)
        print(f'{GM_acc=}')
        grand_mean = np.mean(GM_acc)
        grand_SE = np.std(GM_acc) / np.sqrt(len(GM_acc))
        if age_sn_inc_act.shape[1] == 2:
            grand_t = (grand_mean - 0.5) / grand_SE
        elif age_sn_inc_act.shape[1] == 3:
            grand_t = (grand_mean - 1/3) / grand_SE
        else:
            raise NotImplementedError
        print(f'age {age}: {grand_mean=:.3f} [{grand_SE=:.3f}], {grand_t=:.3f}')
        ts.append(grand_t)
    return ts


# TODO: make a BalanceKFold 50/50 class here

if __name__ == '__main__':
    pass






