import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC

from utils import stdize


def get_many_samples(act_seed_cond, act_tar_cond, n_samples=50, size=7):
    # for run in range(3):
    #     trial_low = run * 38
    #     trial_high = (run + 1) * 38
    #     act_seed_cond[:, trial_low:trial_high] -= \
    #         np.nanmean(act_seed_cond[:, trial_low:trial_high])
    #     act_tar_cond[:, trial_low:trial_high] -= \
    #         np.nanmean(act_tar_cond[:, trial_low:trial_high])

    conn0s = []
    for _ in range(n_samples):
        act_seed_cond0 = act_seed_cond.copy()
        act_tar_cond0 = act_tar_cond.copy()
        for run in range(3):
            trial_low = run * 38
            trial_high = (run + 1) * 38
            non_nans = np.argwhere(~np.isnan(
                act_seed_cond0[0, trial_low:trial_high]))
            non_nans = non_nans.flatten()
            if len(non_nans) < size + 1:
                bad_sn = True
                return [], bad_sn
            non_nan_random = np.random.choice(non_nans,
                                              size=size, #non_nans.shape[0],
                                              replace=False)
            non_nan_random += trial_low
            seed_run = act_seed_cond0[:, non_nan_random]
            tar_run = act_tar_cond0[:, non_nan_random]
            # print(seed_run.shape)

            seed_run = stdize(seed_run, axis=1, nans=True)
            tar_run = stdize(tar_run, axis=1, nans=True)
            conn0 = seed_run[None, ...] * tar_run[:, None, :]

            conn0 = np.nanmean(conn0, axis=-1)
            # print(f'{conn0[0, 1]=}')
            # quit()
            conn0s.append(conn0)
            n_nans = np.sum(np.isnan(conn0))
            n_non_nans = np.sum(~np.isnan(conn0))
            if n_nans > n_non_nans:
                bad_sn = True
                return [], bad_sn
    return conn0s, False


def shuffle_Y_within_subject(Y, groups):
    # Shuffles Y in-place
    group_uniques = np.unique(groups)
    for group in group_uniques:
        idxs = np.argwhere(groups == group).flatten()
        Y_group = Y[idxs]
        np.random.shuffle(Y_group) # in place. can't shuffle Y[idxs] directly
        Y[idxs] = Y_group


def HC_clf(age2idxs, sn_inc_activity_hc, sn_inc_activity_sch, region,
           n_repeats=100, super_sample=False, perm=False):
    all_conns = []
    age_accs = []
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
        sn_inc_activity_sch_age = sn_inc_activity_sch[age_idxs]
        age_conns = []
        num_bads = 0
        for i in range(sn_inc_activity_hc_age.shape[0]):
            act_hc = sn_inc_activity_hc_age[i]
            act_sch = sn_inc_activity_sch_age[i]
            sn_conns = []
            bad_sn = False
            for cond in range(act_hc.shape[0]):

                if super_sample:
                    conn0s, bad_sn = get_many_samples(act_hc[cond], act_sch[cond])
                    if bad_sn:
                        break
                    sn_conns.extend(conn0s)
                else:
                    act_seed_cond = act_hc[cond]
                    # act_seed_cond = stdize(act_seed_cond, axis=1, nans=True)
                    act_tar_cond = act_sch[cond]
                    # act_tar_cond = stdize(act_tar_cond, axis=1, nans=True)
                    for run in range(3):
                        trial_low = run * 38
                        trial_high = (run + 1) * 38
                        act_hc_cond0 = act_seed_cond[:, trial_low:trial_high]
                        act_hc_cond0 = stdize(act_hc_cond0, axis=1, nans=True)
                        act_sch_cond0 = act_tar_cond[:, trial_low:trial_high]
                        act_sch_cond0 = stdize(act_sch_cond0, axis=1, nans=True)
                        conn0 = act_hc_cond0[None, ...] * act_sch_cond0[:, None, :]
                        conn0 = np.nanmean(conn0, axis=-1)
                        n_nans = np.sum(np.isnan(conn0))
                        n_non_nans = np.sum(~np.isnan(conn0))
                        if n_nans > n_non_nans:
                            bad_sn = True
                        sn_conns.append(conn0)
            if bad_sn:
                num_bads += 1
                continue
            age_conns.append(sn_conns)
        age_conns = np.array(age_conns)
        n_sn = age_conns.shape[0]
        if not super_sample:
            for sn in range(n_sn): # PB special (each example is x0 - x1 or vice versa)
                for i in range(3):
                    age_conns[sn, i, :] = age_conns[sn, i, :] - \
                                          age_conns[sn, i + 3, :]
                    age_conns[sn, i + 3, :] = -age_conns[sn, i, :]
        n_cond_ex = age_conns.shape[1] // 2

        X = np.reshape(age_conns, (age_conns.shape[0] * age_conns.shape[1], -1))
        nans = np.isnan(X).any(axis=0)
        X = X[:, ~nans]

        Y = ([0] * n_cond_ex + [1] * n_cond_ex) * n_sn
        Y = np.array(Y)
        # print(f'{len(Y)=}')
        # quit()
        # Y = [0, 0, 0, 1, 1, 1] * n_sn
        groups = np.repeat(np.arange(n_sn), n_cond_ex*2)
        if perm:
            shuffle_Y_within_subject(Y, groups)

        accs = []
        for _ in range(n_repeats):
            grps_unq = np.sort(np.unique(groups))
            grps_unq_ = np.sort(np.unique(groups))
            np.random.shuffle(grps_unq_)
            grp_mapper = {}
            for i, grp in enumerate(grps_unq):
                grp_mapper[grp] = grps_unq_[i]
            groups = np.array([grp_mapper[grp] for grp in groups])
            if len(np.unique(groups)) % 2 == 1:
                rand_group = np.random.choice(groups)
                X_ = X[groups != rand_group]
                Y_ = Y[groups != rand_group]
                groups_ = groups[groups != rand_group]
            else:
                X_ = X
                Y_ = Y
                groups_ = groups
            if super_sample:
                Y_new = []
                X_new = []
                groups_new = []
                for grp in np.unique(groups):
                    for label in [0, 1]:
                        idxs = np.argwhere((groups == grp) &
                                           (Y == label)).flatten()
                        idx = np.random.choice(idxs, size=50)
                        X_new.append(X[idx])
                        Y_new.append(Y[idx])
                        groups_new.append([grp]*len(idx))
                X_ = np.concatenate(X_new)
                Y_ = np.concatenate(Y_new)
                groups_ = np.concatenate(groups_new)
            # print(f'{X_.shape=}')
            # print(f'{Y_.shape=}')
            # print(f'{groups_.shape=}')
            # print(f'{np.unique(groups)=}')
            cv = StratifiedGroupKFold(n_splits=2)
            linear = True
            clf = SVC(kernel='linear' if linear else 'rbf')
            acc = cross_val_score(clf, X_, Y_, cv=cv, groups=groups_)
            acc = np.mean(acc)
            accs.append(acc)
        age2str = {1: 'YA', 2: 'OA'}
        print(f'{region}, age: {age2str[age]} | '
              f'{np.mean(accs)=:.3f} [{np.std(accs)=:.3f}, n = {len(accs)}], '
              f'num bad sn: {num_bads}')
        age_accs.append(np.mean(accs))
    return age_accs


def stratify(X, Y, groups):
    X_new = []
    Y_new = []
    groups_new = []
    for grp in groups.unique():
        n_exs = []
        for cond in [0, 1]:
            n_ex = len(np.argwhere((groups == grp) & (Y == cond)).flatten())
            n_exs.append(n_ex)
        n_ex = min(n_exs)
        X_new.append(X[(groups == grp) & (Y == 0)][:n_ex])
        X_new.append(X[(groups == grp) & (Y == 1)][:n_ex])
        Y_new.append([0] * n_ex + [1] * n_ex)
        groups_new.append([grp] * (n_ex * 2))
    X_new = np.concatenate(X_new)
    Y_new = np.concatenate(Y_new)
    groups_new = np.concatenate(groups_new)
    return X_new, Y_new, groups_new
