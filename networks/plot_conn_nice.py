import os

from tqdm import tqdm

os.chdir('E:\PycharmProjects_E\SchemeRep')

from pathlib import Path

import numpy as np

from atlas_utils import get_atlas
from connRSA.conn_utils import get_BNA_ROIs
from old.network_clf import generic_prep
from old.plot_gen import my_plot_surf
from utils import stdize, run_two_sample_on_2D


def seed_conn_t(age2idxs, sn_inc_activity_seed, sn_inc_activity_sch, kwargs,
                region, accs, atlas):
    all_conns = []
    age2t = {}
    vmin = 1e6
    vmax = -1e6
    ages = [2, 'healthy']
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    for age in ages:
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

    for age in ages:
        dir_out = f'{kwargs["fp"]}_{kwargs["key"]}'
        age2str = {1: 'YA', 2: 'OA', 'healthy': 'Healthy'}
        fn_pic = f'{region.replace(":", "")}_{age2str[age]}.png'
        fp_pic = f'result_pics/seed_conn/{dir_out}/{fn_pic}'
        Path(fp_pic).parent.mkdir(exist_ok=True, parents=True)


        if kwargs['key'] in [('inc', ), 'inc']:
            neg = 'Con'
            pos = 'Inc'
            if kwargs['key_vals'] == (1, 2):
                title_str = f'{region} | {age2str[age]}\n' \
                            f'Congruent (blue) vs. Incongruent (red)'
            else:
                title_str = f'{region} | {age2str[age]}\n' \
                            f'Connectivity ~ congruency'
        elif kwargs['key'] in [('hit_hit', ), ('con_hit', )]:
            neg = 'Miss'
            pos = 'Hit'
            title_str = f'{region} | {age2str[age]}\n' \
                        f'Miss (blue) vs. Hit (red)'
        else:
            raise ValueError

        my_plot_surf(age2t[age], atlas, title_str, fp_out=fp_pic,
                     neg=neg, pos=pos)

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

def make_seed_plots(threshold=0.95, laterality=False, perm=False):
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
    for region in tqdm(regions, desc='Making seed conn plots'):
        if specific_region and (specific_region not in region):
            continue
        idxs = [i for i, ROI in enumerate(ROIs) if region in ROI]
        sn_inc_activity_seed = sn_inc_activity_bna[:, :, idxs, :]

        seed_conn_t(age2idxs, sn_inc_activity_seed, sn_inc_activity_tar, kwargs,
                    region, accs, atlas_tar)

if __name__ == '__main__':
    make_seed_plots()







