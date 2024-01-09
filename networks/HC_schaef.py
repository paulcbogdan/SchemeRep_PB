from pathlib import Path

import numpy as np
from connsearch.report import plot_ROI_scores
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm

from atlas_utils import get_atlas
from conn_utils import get_BNA_ROIs
from network_clf import generic_prep
from utils import stdize

def HC_clf(age2idxs, sn_inc_activity_hc, sn_inc_activity_sch, region,
           n_repeats=100):
    all_conns = []
    age_accs = []
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
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
                for run in range(3):
                    trial_low = run * 38
                    trial_high = (run + 1) * 38
                    act_hc_cond0 = act_hc_cond[:, trial_low:trial_high]
                    act_sch_cond0 = act_sch_cond[:, trial_low:trial_high]
                    conn0 = act_hc_cond0[None, ...] * act_sch_cond0[:, None, :]
                    conn0 = np.nanmean(conn0, axis=-1)
                    sn_conns.append(conn0)
            age_conns.append(sn_conns)
        age_conns = np.array(age_conns)
        age_conns_m = np.nanmean(age_conns, axis=-1)
        age_conns_m = np.concatenate([np.nanmean(age_conns_m[:, :3], axis=1,
                                                 keepdims=True),
                                      np.nanmean(age_conns_m[:, 3:], axis=1,
                                                 keepdims=True)],
                                     axis=1)
        all_conns.append(age_conns_m)
        dif_conn = age_conns_m[:, 0] - age_conns_m[:, 1]
        dif_M = np.nanmean(dif_conn, axis=0)
        dif_std = np.nanstd(dif_conn, axis=0)
        dif_N = np.sum(~np.isnan(dif_conn), axis=0)
        dif_se = dif_std / np.sqrt(dif_N)
        dif_t = dif_M / dif_se
        n_sn = age_conns.shape[0]
        for sn in range(n_sn):
            for i in range(3):
                age_conns[sn, i, :] = age_conns[sn, i, :] - \
                                      age_conns[sn, i + 3, :]
                age_conns[sn, i + 3, :] = -age_conns[sn, i, :]

        X = np.reshape(age_conns, (age_conns.shape[0] * age_conns.shape[1], -1))
        nans = np.isnan(X).any(axis=0)
        X = X[:, ~nans]

        Y = [0, 0, 0, 1, 1, 1] * n_sn
        groups = np.repeat(np.arange(n_sn), 6)

        accs = []
        for _ in tqdm(range(n_repeats)):
            grps_unq = np.sort(np.unique(groups))
            grps_unq_ = np.sort(np.unique(groups))
            np.random.shuffle(grps_unq_)
            grp_mapper = {}
            for i, grp in enumerate(grps_unq):
                grp_mapper[grp] = grps_unq_[i]
            groups = np.array([grp_mapper[grp] for grp in groups])
            cv = StratifiedGroupKFold(n_splits=3)
            linear = True
            clf = SVC(kernel='linear' if linear else 'rbf')
            acc = cross_val_score(clf, X, Y, cv=cv, groups=groups)
            acc = np.mean(acc)
            accs.append(acc)
        age2str = {1: 'YA', 2: 'OA'}
        print(f'{region}, age: {age2str[age]} | '
              f'{np.mean(accs)=:.3f} [{np.std(accs)=:.3f}]')
        age_accs.append(np.mean(accs))
    return age_accs

def HC_t(age2idxs, sn_inc_activity_hc, sn_inc_activity_sch, kwargs, region,
         accs, atlas):
    all_conns = []
    age2t = {}
    vmin = 1e6
    vmax = -1e6
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        print(f'{len(age_idxs)=}')
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
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

        all_conns.append(age_conns_m)
        dif_conn = age_conns_m[:, 0] - age_conns_m[:, 1]
        dif_M = np.nanmean(dif_conn, axis=0)
        dif_std = np.nanstd(dif_conn, axis=0)
        dif_N = np.sum(~np.isnan(dif_conn), axis=0)
        dif_se = dif_std / np.sqrt(dif_N)
        dif_t = dif_M / dif_se
        age2t[age] = dif_t
        vmin = min(vmin, np.nanquantile(dif_t, 0.025))
        vmax = max(vmax, np.nanquantile(dif_t, 0.975))
        print(f'Test: {np.nanquantile(dif_t, 0.975)}')


    for age in [1, 2]:
        coords = atlas['coords']
        dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
        age2str = {1: 'YA', 2: 'OA'}
        fp_pic = f'mass_ttest/{dir_out}/{region}_{age2str[age]}.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        acc = accs[region][age-1]
        plot_ROI_scores(age2t[age], coords, fp_out=fp_pic, show=True,
                        title=f'{region} | {age2str[age]}, {kwargs["key"]}: '
                              f'Acc = {acc:.1%}',
                        vmin=vmin, vmax=vmax)

    dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
    fp_pic = f'mass_ttest/{dir_out}/{region}_interaction.png'
    itr_t = run_two_sample_on_2D(all_conns[0][:, 0] - all_conns[0][:, 1],
                                 all_conns[1][:, 0] - all_conns[1][:, 1])
    plot_ROI_scores(itr_t, coords, fp_out=fp_pic, show=True,
                    title=f'{region} | age x {kwargs["key"]}')

    fp_pic = f'mass_ttest/{dir_out}/{region}_age_eff.png'
    itr_t = run_two_sample_on_2D(all_conns[0][:, 0] + all_conns[0][:, 1],
                                 all_conns[1][:, 0] + all_conns[1][:, 1])
    plot_ROI_scores(itr_t, coords, fp_out=fp_pic, show=True,
                    title=f'{region} | main effect of age')

def plot_M(age2idxs, sn_inc_activity_hc, sn_inc_activity_sch, kwargs, region,
           atlas):
    age2gm = {}
    vmin = 1e6
    vmax = -1e6
    for age in [1, 2]:
        age_idxs = age2idxs[age]
        print(f'{len(age_idxs)=}')
        sn_inc_activity_hc_age = sn_inc_activity_hc[age_idxs]
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
        vmin = min(vmin, np.nanquantile(age_GM, 0.025))
        vmax = max(vmax, np.nanquantile(age_GM, 0.975))

    coords = atlas['coords']
    dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
    age2str = {0: 'YA & OA', 1: 'YA', 2: 'OA'}
    GGM = np.nanmean([age2gm[1], age2gm[2]], axis=0)
    age2gm[0] = GGM
    for age in [0, 1, 2]:
        fp_pic = f'mass_M/{dir_out}/{region}_{age2str[age]}.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        fp_pic = f'mass_M/{dir_out}/{region}_{age2str[age]}.png'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)
        plot_ROI_scores(list(age2gm[age]), coords, fp_out=fp_pic, show=True,
                        title=f'{region} | {age2str[age]}',
                        vmin=vmin, vmax=vmax, cmap='viridis')


def run_HC_schaef(threshold=0.95):
    fp = 'obj4_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'atlas_name': 'schaefer',
              'key_vals': (1, 3),
              'odd_even': False,
              }
    # kwargs = {'fp': fp, 'split': False,
    #           'key': 'vis_hit',
    #           'atlas_name': 'schaefer',
    #           'key_vals': (False, True),
    #           'odd_even': False,
    #           }
    partitions, sn_inc_activity_sch, age2idxs, top_edges_mat, i2name = \
        generic_prep(kwargs, threshold=threshold)

    kwargs1 = kwargs.copy()
    kwargs1['atlas_name'] = 'BNA'
    _, sn_inc_activity_bna, _, _, _ = generic_prep(kwargs1,
                                                   threshold=threshold)
    print(sn_inc_activity_bna.shape)
    BNA = get_atlas()
    ROIs = get_BNA_ROIs()
    regions = BNA['tick_labels']
    accs = {}
    atlas_sf = get_atlas(schaefer=True)

    for region in regions:
        # if region != 'LOC':
        #     continue

        idxs = [i for i, ROI in enumerate(ROIs) if region in ROI]
        sn_inc_activity_rg = sn_inc_activity_bna[:, :, idxs, :]
        # plot_M(age2idxs, sn_inc_activity_rg, sn_inc_activity_sch, kwargs,
        #        region, atlas_sf)
        # continue
        accs_rg = HC_clf(age2idxs, sn_inc_activity_rg, sn_inc_activity_sch,
                         region, n_repeats=10)
        accs[region] = accs_rg
        HC_t(age2idxs, sn_inc_activity_rg, sn_inc_activity_sch, kwargs,
             region, accs, atlas_sf)
    quit()



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







