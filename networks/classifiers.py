import numpy as np
import numpy.ma as ma
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm


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
        print(f'Number dropped to stratify: '
              f'{n_pre} - {n_post} = {n_pre - n_post}')
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




def partition_classifier(sn_inc_activity, age2idxs, edges,
                         stratification_strategy=2,
                         trial_mapper=None):
    # If you group by run, the data must be sorted properly
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
                for run in range(3):
                    if empty_mapper:
                        run_low = run * 38
                        run_high = (run + 1) * 38
                        trial_set = act0[run_low:run_high]
                        trial_set = abs(trial_set[:, None, :] -
                                        trial_set[:, :, None])
                        trial_mapper[(age, sn, inc0, run)] = trial_set
                    else:
                        trial_set = trial_mapper[(age, sn, inc0, run)]
                    if edges:
                        edges_ = zip(*edges)
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
            try:
                X = np.concatenate(X, axis=0)
            except ValueError:
                print(f'Fail not enough data: {age=}, {sn=}')
                continue

            nan_edges = np.any(np.isnan(X), axis=0)
            n_nan_edges = np.sum(nan_edges)

            if len(X.shape) < 2 or X.shape[1] < 50:#n_nan_edges > X.shape[1] // 2:
                print(f'Fail too many nans ({n_nan_edges}): {age=}, {sn=} | '
                      f'{X.shape=}')
                return [np.nan, np.nan]

            X = X[:, ~nan_edges]
            # print(f'{np.sum(nan_edges)=}')

            Y = np.array(Y)
            # print(f'{Y=}')
            # print(f'{len(Y)=}')
            # print(f'{groups=}')
            if stratification_strategy == 1:
                X, Y, groups = stratify(X, Y, groups=groups)
                if len(np.unique(groups)) != 3:
                    print(f'Not enough data ({age=}, {sn=})')
                    continue
                cv = StratifiedGroupKFold(n_splits=3)
                clf = SVC(kernel='rbf')
                acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
                acc = np.mean(acc)
            elif stratification_strategy in [2, 3]:
                for g in range(3):
                    X[groups == g] -= np.mean(X[groups == g], axis=0)
                # print(f'{len(Y)=}')

                repeats = 10 if stratification_strategy == 2 else 100
                cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=repeats)
                clf = SVC(kernel='rbf')
                acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
                acc = np.mean(acc)
                # print(f'{acc=:.3f}')
                # print('toast')
                # print(X.shape)
                # quit()
                # print(groups)
                # quit()


            # print(f'{age=}, {sn=}: {acc=}')
            # print(f'{acc=:.3f}')
            GM_acc.append(acc)
            # quit()
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






