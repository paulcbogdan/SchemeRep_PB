import os
os.chdir('E:\PycharmProjects_E\SchemeRep')

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC

from atlas_utils import get_atlas
from connRSA.conn_utils import get_BNA_ROIs
from old.network_clf import generic_prep
from old.plot_gen import my_plot_surf
from utils import stdize
# from connsearch import print_list_stats

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


def HC_t(age2idxs, sn_inc_activity_seed, sn_inc_activity_sch, kwargs, region,
         accs, atlas):
    all_conns = []
    age2t = {}
    vmin = 1e6
    vmax = -1e6
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    for age in ['healthy']:
        age_idxs = age2idxs[age]
        seed_age = sn_inc_activity_seed[age_idxs]
        seed_age = np.nanmean(seed_age, axis=2)
        tar_age = sn_inc_activity_sch[age_idxs]
        age_conns = []
        for i in range(seed_age.shape[0]):
            seed_age_sn = seed_age[i]
            tar_age_sn = tar_age[i]
            sn_conns = []

            for cond in range(seed_age_sn.shape[0]):
                seed_age_sn_cond = seed_age_sn[cond]
                tar_age_sn_cond = tar_age_sn[cond]
                tar_age_sn_cond = stdize(tar_age_sn_cond, axis=1, nans=True)
                conn = seed_age_sn_cond[None, :] * tar_age_sn_cond
                conn = np.nanmean(conn, axis=-1)
                sn_conns.append(conn)
            age_conns.append(sn_conns)
        age_conns = np.array(age_conns)
        all_conns.append(age_conns)
        dif_conn = age_conns[:, 0] - age_conns[:, 2]
        dif_M = np.nanmean(dif_conn, axis=0)
        dif_std = np.nanstd(dif_conn, axis=0)
        dif_N = np.sum(~np.isnan(dif_conn), axis=0)
        dif_se = dif_std / np.sqrt(dif_N)
        dif_t = dif_M / dif_se
        age2t[age] = dif_t
        vmin = min(vmin, np.nanquantile(dif_t, 0.025))
        vmax = max(vmax, np.nanquantile(dif_t, 0.975))

    for age in [1, 2]:
        coords = atlas['coords']
        dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
        age2str = {1: 'YA', 2: 'OA'}
        fn_pic = f'{region.replace(":", "")}_{age2str[age]}.png'
        fp_pic = f'results_pics/seed_conn/{dir_out}/{fn_pic}'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        acc = accs[region][age-1]
        if acc is None:
            acc_str = ''
        else:
            acc_str = f' (Acc = {acc:.1%})'

                    # f'Acc = {acc:.1%}'
        # plot_ROI_scores(age2t[age], coords, fp_out=fp_pic, show=False,
        #                 title=title_str, vmin=vmin, vmax=vmax)
        if kwargs['key'] in [('inc', ), 'inc']:
            neg = 'Con'
            pos = 'Inc'
            title_str = f'{region} | {age2str[age]}{acc_str}\n' \
                        f'Congruent (blue) vs. Incongruent (red)'
        elif kwargs['key'] in [('hit_hit', ), ('con_hit', )]:
            neg = 'Miss'
            pos = 'Hit'
            title_str = f'{region} | {age2str[age]}{acc_str}\n' \
                        f'Miss (blue) vs. Hit (red)'


        my_plot_surf(age2t[age], atlas, title_str, fp_out=fp_pic,
                     neg=neg, pos=pos)

    if kwargs['key'] in [('inc', ), 'inc']:
        itr_title = f'{region}\nAge x Congruency'
        neg = 'OA\n↑Con'
        pos = 'OA\n↑Inc'
    elif kwargs['key'] in [('hit_hit', ), ('con_hit', )]:
        itr_title = f'{region}\nAge x Hit-Hit'
        neg = 'OA\n↑Miss'
        pos = 'OA\n↑Hit'

    dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
    fp_pic = (f'results_pics/seed_conn/{dir_out}'
              f'/{region.replace(":", "")}_interaction.png')
    itr_t = run_two_sample_on_2D(all_conns[0][:, 0] - all_conns[0][:, 1],
                                 all_conns[1][:, 0] - all_conns[1][:, 1])
    # plot_ROI_scores(itr_t, coords, fp_out=fp_pic, show=False,
    #                 title=f'{region} | age x {kwargs["key"]}')

    my_plot_surf(-itr_t, atlas, itr_title,
                 fp_out=fp_pic, neg=neg, pos=pos)

    fp_pic = (f'results_pics/seed_conn/{dir_out}'
              f'/{region.replace(":", "")}_age_eff.png')
    age_t = run_two_sample_on_2D(all_conns[0][:, 0] + all_conns[0][:, 1],
                                 all_conns[1][:, 0] + all_conns[1][:, 1])
    # plot_ROI_scores(age_t, coords, fp_out=fp_pic, show=False,
    #                 title=f'{region} | main effect of age')
    my_plot_surf(age_t, atlas, f'{region.replace(":", "")}\nMain effect of age',
                 fp_out=fp_pic, neg='OA', pos='YA')

def plot_M(age2idxs, sn_inc_activity_hc, sn_inc_activity_sch, kwargs, region,
           atlas):
    age2gm = {}
    vmin = 1e6
    vmax = -1e6
    age2cond0m = {}
    age2cond1m = {}
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
        # sn_inc_activity_hc_age = sn_inc_activity_hc_age[:, :, [0], :]
        sn_inc_activity_sch_age = sn_inc_activity_sch[age_idxs]
        age_conns = []
        for i in range(sn_inc_activity_hc_age.shape[0]):
            act_hc = sn_inc_activity_hc_age[i]
            act_sch = sn_inc_activity_sch_age[i]
            sn_conns = []
            for cond in range(act_hc.shape[0]):
                act_hc_cond = act_hc[cond]
                act_hc_cond = stdize(act_hc_cond, axis=1, nans=True)
                act_sch_cond = act_sch[cond]
                act_sch_cond = stdize(act_sch_cond, axis=1, nans=True)
                conn = act_hc_cond[None, ...] * act_sch_cond[:, None, :]
                conn = np.nanmean(conn, axis=-1)
                sn_conns.append(conn)
            age_conns.append(sn_conns)
        age_conns = np.array(age_conns)
        age_conns_m = np.nanmean(age_conns, axis=-1)
        age_GM = np.nanmean(age_conns_m, axis=(0, 1))
        age2gm[age] = age_GM
        age2cond0m[age] = np.nanmean(age_conns_m[:, 0], axis=0)
        age2cond1m[age] = np.nanmean(age_conns_m[:, 1], axis=0)
        vmin = min(vmin, np.nanquantile(age_GM, 0.025))
        vmax = max(vmax, np.nanquantile(age_GM, 0.9))

    coords = atlas['coords']
    dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
    age2str = {0: 'YA & OA', 1: 'YA', 2: 'OA'}
    GGM = np.nanmean([age2gm[1], age2gm[2]], axis=0)
    age2gm[0] = GGM
    atlas_name = atlas['name']
    for age in [0, 1, 2]:
        fp_pic = f'mass_M/{dir_out}_{atlas_name}/{region}_{age2str[age]}.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        # plot_ROI_scores(list(age2gm[age]), coords, fp_out=fp_pic, show=False,
        #                 title=f'{region} | {age2str[age]}, Inc & Con',
        #                 vmin=vmin, vmax=vmax, cmap='viridis')
        my_plot_surf(np.array(list(age2gm[age])), atlas,
                     f'{region} | {age2str[age]}, Inc & Con',
                     thresh=0.1, vmax=vmax,
                     fp_out=fp_pic)
        if age == 0:
            continue

        fp_pic = f'mass_M/{dir_out}_{atlas_name}/' \
                 f'{region}_{age2str[age]}_cond0.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        # plot_ROI_scores(list(age2cond0m[age]), coords, fp_out=fp_pic, show=False,
        #                 title=f'{region} | {age2str[age]}, Inc',
        #                 vmin=vmin, vmax=vmax, cmap='viridis')
        my_plot_surf(np.array(list(age2cond0m[age])), atlas,
                     f'{region} | {age2str[age]}, Inc',
                     thresh=0.1, vmax=vmax,
                     fp_out=fp_pic)

        fp_pic = f'mass_M/{dir_out}_{atlas_name}/' \
                 f'{region}_{age2str[age]}_cond1.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        # plot_ROI_scores(list(age2cond1m[age]), coords, fp_out=fp_pic, show=False,
        #                 title=f'{region} | {age2str[age]}, Con',
        #                 vmin=vmin, vmax=vmax, cmap='viridis')
        my_plot_surf(np.array(list(age2cond1m[age])), atlas,
                     f'{region} | {age2str[age]}, Con',
                     thresh=0.1, vmax=vmax,
                     fp_out=fp_pic)
    print(f'Plotted M: {region=}')

def get_keep_idxs(combine_regions=False):
    atlas = get_atlas(combine_regions=combine_regions)
    rois = atlas['ROIs']
    keep_idxs = list(range(len(rois)))
    keep_regions = atlas['tick_labels']
    bad_regions = ['Tha', 'Str']
    drop_idxs = []

    for idx in list(keep_idxs):
        name = rois[idx]
        if not any(region in name for region in keep_regions):
            keep_idxs.remove(idx)
            drop_idxs.append(idx)
            continue
        if any(region in name for region in bad_regions):
            keep_idxs.remove(idx)
            drop_idxs.append(idx)
            continue
    keep_idxs = sorted(list(set(keep_idxs)))
    drop_idxs = sorted(list(set(drop_idxs)))
    return keep_idxs, drop_idxs

def run_HC_schaef(threshold=0.95, laterality=False, perm=False):
    fp = 'obj7_fMRI'
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              }

    partitions, sn_inc_activity_tar, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)
    keep_idxs, drop_idxs = get_keep_idxs()
    sn_inc_activity_tar[:, :, drop_idxs, :] = np.nan

    kwargs['atlas_name'] = 'BNA'
    kwargs1 = kwargs.copy()
    _, sn_inc_activity_bna, _, _, _ = generic_prep(kwargs1,
                                                   threshold=threshold)

    BNA = get_atlas()
    ROIs = get_BNA_ROIs()
    regions = BNA['tick_labels']
    if laterality:
        regions = [f'{r}_{lr}' for r in regions for lr in ['L', 'R']]

    accs = {}
    atlas_tar = get_atlas(schaefer=kwargs['atlas_name'] == 'schaefer')
    atlas_tar['name'] = kwargs['atlas_name']
    # atlas_tar['coords'] = [atlas_tar['coords'][i] for i in keep_idxs]

    specific_region = None
    for region in regions:
        if specific_region and (specific_region not in region):
            continue
        idxs = [i for i, ROI in enumerate(ROIs) if region in ROI]
        sn_inc_activity_seed = sn_inc_activity_bna[:, :, idxs, :]
        if perm:
            OA_accs = []
            YA_accs = []
            for nsim in range(100):
                print(f'PERM: {region} | {nsim}')
                YA_acc, OA_acc = HC_clf(age2idxs, sn_inc_activity_seed,
                                        sn_inc_activity_tar,
                                        region, n_repeats=10, perm=True,
                                        super_sample=True,)
                OA_accs.append(OA_acc)
                YA_accs.append(YA_acc)
                if nsim % 5 == 0:
                    print_list_stats(OA_accs)
                    print_list_stats(YA_accs)
            continue
        else:
            accs_rg = 0
            # accs_rg = HC_clf(age2idxs, sn_inc_activity_seed, sn_inc_activity_tar,
            #                  region, n_repeats=10, super_sample=True)
        accs[region] = accs_rg
        HC_t(age2idxs, sn_inc_activity_seed, sn_inc_activity_tar, kwargs,
             region, accs, atlas_tar)
    quit()

# Number of items: 96
# Mean item: 0.500
# Median item: 0.500
# Min item: 0.494
# Max item: 0.511
# p(under 50%): 0.521 | p(above 50%): 0.479
# Percentile: Accuracy | 1.0: 0.4939, 0.75: 0.4976, 0.5: 0.4998, 0.25: 0.5027, 0.1: 0.5040, 0.05: 0.5049, 0.01: 0.5108, 0.005: 0.5108, 0.001: 0.5108


def run_two_sample_on_2D(ar0, ar1):
    YA_M = np.nanmean(ar0, axis=0)
    OA_M = np.nanmean(ar1, axis=0)
    YA_sd = np.nanstd(ar0, axis=0)
    OA_sd = np.nanstd(ar1, axis=0)
    YA_N = np.sum(~np.isnan(ar0), axis=0)
    OA_N = np.sum(~np.isnan(ar1), axis=0)

    both_sd = ((YA_sd ** 2) * (YA_N - 1) +
               (OA_sd ** 2) * (OA_N - 1)) / \
              (YA_N + OA_N - 2)
    both_se = np.sqrt(both_sd * (1 / YA_N + 1 / OA_N))
    t = (YA_M - OA_M) / both_se
    t[np.isnan(t)] = 0
    return t

if __name__ == '__main__':
    run_HC_schaef()







