import itertools
import pickle
import random
from pathlib import Path
from time import time

import numpy as np

from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
from Study2A.rs_connectivity_funcs import get_hemi_ps
from Study2B.analyze_Study2B import get_final_HCP_sns, get_HCP_task, get_HCP_rs_sns, ttest_on_correlations
from Utils.atlas_funcs import get_atlas


def get_quads_schaefer(schaefer=(True, 400), combine_regions=False):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                             anat_ver=3, combine_regions=combine_regions,
                             schaefer=schaefer)

    ps_hemi = get_hemi_ps(p_d_ant, p_d_pos, p_v_ant, p_v_pos, schaefer=schaefer,
                          combine_regions=combine_regions)

    Lda, Rda, Ldp, Rdp, Lva, Rva, Lvp, Rvp = (
        ps_hemi['Lda'], ps_hemi['Rda'], ps_hemi['Ldp'], ps_hemi['Rdp'], \
        ps_hemi['Lva'], ps_hemi['Rva'], ps_hemi['Lvp'], ps_hemi['Rvp'])
    # print(F'{len(Lda)=}, {len(Rda)=}, {len(Ldp)=}, {len(Rdp)=}, '
    #       f'{len(Lva)=}, {len(Rva)=}, {len(Lvp)=}, {len(Rvp)=}')
    # quit()

    if not combine_regions:
        Rda = get_closest_pair(Lda, Rda, combine_regions=combine_regions,
                               schaefer=schaefer)
        Rdp = get_closest_pair(Ldp, Rdp, combine_regions=combine_regions,
                                schaefer=schaefer)
        Rva = get_closest_pair(Lva, Rva, combine_regions=combine_regions,
                                schaefer=schaefer)
        Rvp = get_closest_pair(Lvp, Rvp, combine_regions=combine_regions,
                                schaefer=schaefer)

    return Lda, Rda, Ldp, Rdp, Lva, Rva, Lvp, Rvp
    # print(ps_hemi)
    # quit()
    # return ps_hemi['Lda'], ps_hemi['Rda'], ps_hemi['Ldp'], ps_hemi['Rdp'], \
    #     ps_hemi['Lva'], ps_hemi['Rva'], ps_hemi['Lvp'], ps_hemi['Rvp']


def get_closest_pair(p_L, p_R, combine_regions=False, schaefer=(True, 400)):
    atlas = get_atlas(combine_regions=combine_regions,
                      schaefer=schaefer)
    p_R_new = []
    for i in p_L:
        coord_i = atlas['coords'][i]
        coord_i_flipp = np.array([-coord_i[0], coord_i[1], coord_i[2]])
        min_dist_j = 1e6
        min_j = None
        for j in p_R:
            coord_j = np.array(list(atlas['coords'][j]))
            dist = np.linalg.norm(coord_i_flipp - coord_j)
            if dist < min_dist_j:
                min_dist_j = dist
                min_j = j
        p_R_new.append(min_j)
        assert min_j is not None
    return p_R_new
        # print(f'{i=}, {min_j=}, {min_dist_j=}')


def run_analysis_Study2B_sch(num_test=500, skip_other=True,
                             combine_regions=False, focus='combo',
                             half=None, just_lr=None):
    sns = get_final_HCP_sns()#[:25]
    if half is not None:
        half_str = f'_h{half}'
        if half == 0:
            sns = sns[:len(sns) // 2]
        elif half == 1:
            sns = sns[len(sns) // 2:]
        else:
            raise ValueError
    else:
        half_str = ''
    if just_lr is not None:
        lr_str = f'_{just_lr}'
    else:
        lr_str = ''


    schaefer =  ('schaefer', 100)
    Lda, Rda, Ldp, Rdp, Lva, Rva, Lvp, Rvp = (
        get_quads_schaefer(combine_regions=combine_regions,
                           schaefer=schaefer))

    combine_regions = (combine_regions, schaefer)

    # da_len = min(len(Lda), len(Rda))
    # dp_len = min(len(Ldp), len(Rdp))
    # va_len = min(len(Lva), len(Rva))
    # vp_len = min(len(Lvp), len(Rvp))

    if just_lr == 'L':
        p_d_ant_all = Lda
        p_d_pos_all = Ldp
        p_v_ant_all = Lva
        p_v_pos_all = Lvp
        da_len = len(Lda)
        dp_len = len(Ldp)
        va_len = len(Lva)
        vp_len = len(Lvp)
    elif just_lr == 'R':
        p_d_ant_all = Rda
        p_d_pos_all = Rdp
        p_v_ant_all = Rva
        p_v_pos_all = Rvp
        da_len = len(Rda)
        dp_len = len(Rdp)
        va_len = len(Rva)
        vp_len = len(Rvp)
    else:
        p_d_ant_all = Lda + Rda
        p_d_pos_all = Ldp + Rdp
        p_v_ant_all = Lva + Rva
        p_v_pos_all = Lvp + Rvp
        da_len = min(len(Lda), len(Rda))
        dp_len = min(len(Ldp), len(Rdp))
        va_len = min(len(Lva), len(Rva))
        vp_len = min(len(Lvp), len(Rvp))

    combos = itertools.product(list(range(da_len)), list(range(dp_len)),
                               list(range(va_len)), list(range(vp_len)))
    combos = list(combos)
    # print(combos[:100])
    np.random.seed(0)
    np.random.shuffle(combos)
    # print(combos[:100])
    # quit()
    print(F'{len(combos)=} | {num_test=}')

    combos = combos[:num_test]
    # print(len(combos))
    # print(F'{len(combos)=}')
    # quit()

    rs_efs_l = []
    task_efs_l = []
    reliability_l = []
    reliability_task_l = []
    for (a, b, c, d) in combos:
        if just_lr == 'L':
            pda_i = [Lda[a]]
            pdp_i = [Ldp[b]]
            pva_i = [Lva[c]]
            pvp_i = [Lvp[d]]
        elif just_lr == 'R':
            pda_i = [Rda[a]]
            pdp_i = [Rdp[b]]
            pva_i = [Rva[c]]
            pvp_i = [Rvp[d]]
        else:
            pda_i = [Lda[a], Rda[a]]
            pdp_i = [Ldp[b], Rdp[b]]
            pva_i = [Lva[c], Rva[c]]
            pvp_i = [Lvp[d], Rvp[d]]
        print(f'{pda_i=}, {pdp_i=}, {pva_i=}, {pvp_i=}')

        t_st = time()

        rs_efs1 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                 lr='LR', combine_regions=combine_regions, bilateral=False, )
        rs_efs2 = get_HCP_rs_sns(sns, pda_i, pdp_i, pva_i, pvp_i,
                                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                 lr='RL', combine_regions=combine_regions, bilateral=False, )
        rs_efs = np.nanmean([rs_efs1, rs_efs2], axis=0)
        print(f'Resting time: {time() - t_st:.5f} s')

        t_st = time()
        task_efs, task_efs_lr, task_efs_rl = (
            get_HCP_task(sns, pda_i, pdp_i, pva_i, pvp_i,
                         combine_regions=combine_regions,
                         focus=focus))
        print(f'\tTask time ({focus}): {time() - t_st:.5f} s')

        num_nans = np.sum(np.isnan(rs_efs))
        assert num_nans == 0, f'{num_nans=}, f{rs_efs.shape=}'

        task_efs_l.append(task_efs)
        rs_efs_l.append(rs_efs)

        if len(task_efs_l) % 10 == 0:
            # print(f'{len(task_efs_l)=}, {len(rs_efs_l)=}')
            ttest_on_correlations(task_efs_l, rs_efs_l)

            str_shape = '_' + str(np.array(task_efs_l).shape)
            fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_task_{focus}_schaefer{half_str}{lr_str}.pkl'
            Path(fp_pkl).parent.mkdir(parents=True, exist_ok=True)
            with open(fp_pkl, 'wb') as f:
                pickle.dump(task_efs_l, f)
            fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_rs_{focus}_schaefer{half_str}{lr_str}.pkl'
            with open(fp_pkl, 'wb') as f:
                pickle.dump(rs_efs_l, f)

    ttest_on_correlations(task_efs_l, rs_efs_l)

    str_shape = '_' + str(np.array(task_efs_l).shape)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_task_{focus}_schaefer{half_str}{lr_str}.pkl'
    Path(fp_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(fp_pkl, 'wb') as f:
        pickle.dump(task_efs_l, f)
    fp_pkl = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr_{str_shape}_rs_{focus}_schaefer{half_str}{lr_str}.pkl'
    with open(fp_pkl, 'wb') as f:
        pickle.dump(rs_efs_l, f)


if __name__ == '__main__':
    run_analysis_Study2B_sch(half=None)
    # run_analysis_Study2B_sch(half=0)#, focus='loss')
    # run_analysis_Study2B_sch(half=1)#, focus='loss')

    # run_analysis_Study2B_sch(half=None)
    # run_analysis_Study2B_sch(half=None, just_lr='L')
    # run_analysis_Study2B_sch(half=None, just_lr='R')

