from time import time

import numpy as np

from Study2B.analyze_Study2B import get_final_HCP_sns, get_quads, get_rs_conn_sns


def consistency_overall_rs_mag():
    pass


def get_rs_edge_tvc(sns, i, j,
                    all_pda, all_pdp, all_pva, all_pvp, lr='LR',
                    combine_regions=False, bilateral=False,
                    reg_global=False, no_compcor=False,
                    ix=True
                    ):
    kw = {'lr': lr, 'combine_regions': combine_regions,
          'bilateral': bilateral, 'reg_global': reg_global,
          'no_compcor': no_compcor, 'rs': True}
    # t_st = time()
    rs_conn, map2new = get_rs_conn_sns(tuple(sns),
                                       tuple(all_pda), tuple(all_pdp),
                                       tuple(all_pva), tuple(all_pvp), **kw)

    if i > j:
        i0 = j
        j0 = i
    else:
        i0 = i
        j0 = j

    i, j = map2new[i0], map2new[j0]
    tvc = rs_conn[:, i, j, :]
    return tvc


def get_network_triangle_consistency(sns, p0, p1, p2, p_d_ant_all, p_d_pos_all,
                                     p_v_ant_all, p_v_pos_all,
                                     combine_regions=False):
    sns_all_r_lr = []
    sns_all_r_rl = []
    for i in p0:
        for j in p1:
            for k in p2:


                tvc_ij_lr = get_rs_edge_tvc(sns, i, j,
                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                            lr='LR', combine_regions=combine_regions)


                tvc_ik_lr = get_rs_edge_tvc(sns, i, k,
                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                            lr='LR', combine_regions=combine_regions)


                fluc_lr = np.nanmean(np.abs(tvc_ij_lr - tvc_ik_lr), axis=1)


                sns_all_r_lr.append(fluc_lr)
                tvc_ij_rl = get_rs_edge_tvc(sns, i, j,
                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                            lr='RL', combine_regions=combine_regions)
                tvc_ik_rl = get_rs_edge_tvc(sns, i, k,
                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                            lr='RL', combine_regions=combine_regions)
                # print(f'{tvc_ij_rl.shape=}')
                # quit()
                fluc_rl = np.nanmean(np.abs(tvc_ij_rl - tvc_ik_rl), axis=1)
                sns_all_r_rl.append(fluc_rl)


    sns_all_r_lr = np.array(sns_all_r_lr)
    sns_all_r_rl = np.array(sns_all_r_rl)


    # sns_all_r_lr -= np.nanmean(sns_all_r_lr, axis=1, keepdims=True)
    # sns_all_r_rl -= np.nanmean(sns_all_r_rl, axis=1, keepdims=True)

    sn_overall_lr = np.nanmean(sns_all_r_lr, axis=0)
    sn_overall_rl = np.nanmean(sns_all_r_rl, axis=0)
    # print(sns_all_r_rl.shape)
    # quit()
    icc_overall = np.corrcoef(sn_overall_lr, sn_overall_rl)[0, 1]

    # roi_overall_lr = np.nanmean(sns_all_r_lr, axis=1)
    # roi_overall_rl = np.nanmean(sns_all_r_rl, axis=1)

    within_subj_icc_l = []
    for sn_j in range(sns_all_r_lr.shape[1]):
        sns_r_lr = sns_all_r_lr[:, sn_j]
        sns_r_rl = sns_all_r_rl[:, sn_j]
        # sns_r_rl = roi_overall_lr

        icc = np.corrcoef(sns_r_lr, sns_r_rl)[0, 1]
        within_subj_icc_l.append(icc)

    between_subj_icc_l = []
    for edge_i in range(sns_all_r_lr.shape[0]):
        sns_r_lr = sns_all_r_lr[edge_i, :]
        sns_r_rl = sns_all_r_rl[edge_i, :]
        icc = np.corrcoef(sns_r_lr, sns_r_rl)[0, 1]
        between_subj_icc_l.append(icc)

    within_subj_icc_l = np.array(within_subj_icc_l)
    between_subj_icc_l = np.array(between_subj_icc_l)
    assert np.isnan(within_subj_icc_l).sum() == 0, 'Percentage nan: {:.1%}'.format(np.isnan(within_subj_icc_l).mean())
    assert np.isnan(between_subj_icc_l).sum() == 0, 'Percentage nan: {:.1%}'.format(np.isnan(between_subj_icc_l).mean())
    M_icc_within = np.nanmean(within_subj_icc_l)
    M_icc_between = np.nanmean(between_subj_icc_l)

    return M_icc_within, M_icc_between, icc_overall

def get_network_connesitency(sns, p0, p1, p_d_ant_all, p_d_pos_all,
                             p_v_ant_all, p_v_pos_all,
                             combine_regions=False, subtract_M=False):
    sns_all_r_lr = []
    sns_all_r_rl = []
    for i in p0:
        for j in p1:
            # i = 14
            # j = 134
            sns_r_lr = get_rs_edge_tvc(sns, i, j,
                                       p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                       lr='LR', combine_regions=combine_regions)
            # print(f'{sns_r_lr=}')
            # print(f'{i}, {j}')
            # quit()
            sns_r_lr = np.nanmean(sns_r_lr, axis=1)
            sns_all_r_lr.append(sns_r_lr)
            sns_r_rl = get_rs_edge_tvc(sns, i, j,
                                       p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                       lr='RL', combine_regions=combine_regions)
            sns_r_rl = np.nanmean(sns_r_rl, axis=1)
            sns_all_r_rl.append(sns_r_rl)
            # print(f'{i=}, {j=}, {sns_r_lr=}, {sns_r_rl=}')
            # quit()
    sns_all_r_lr = np.array(sns_all_r_lr)
    sns_all_r_rl = np.array(sns_all_r_rl)
    # print(np.nanmean(sns_all_r_lr, axis=1).shape)
    # print(sns_all_r_rl.shape)
    # quit()

    if subtract_M:
        roi_overall_lr = np.nanmean(sns_all_r_lr, axis=1, keepdims=True)
        roi_overall_rl = np.nanmean(sns_all_r_rl, axis=1, keepdims=True)
        sns_all_r_lr -= roi_overall_lr
        sns_all_r_rl -= roi_overall_rl

    sn_overall_lr = np.nanmean(sns_all_r_lr, axis=0)
    sn_overall_rl = np.nanmean(sns_all_r_rl, axis=0)
    icc_overall = np.corrcoef(sn_overall_lr, sn_overall_rl)[0, 1]

    within_subj_icc_l = []
    for sn_j in range(sns_all_r_lr.shape[1]):
        sns_r_lr = sns_all_r_lr[:, sn_j]
        # print(roi_overall_rl.shape)
        # quit()
        # sns_r_rl = roi_overall_rl[:, 0]
        sns_r_rl = sns_all_r_rl[:, sn_j]
        icc = np.corrcoef(sns_r_lr, sns_r_rl)[0, 1]
        within_subj_icc_l.append(icc)

    between_subj_icc_l = []
    for edge_i in range(sns_all_r_lr.shape[0]):
        sns_r_lr = sns_all_r_lr[edge_i, :]
        sns_r_rl = sns_all_r_rl[edge_i, :]
        icc = np.corrcoef(sns_r_lr, sns_r_rl)[0, 1]
        between_subj_icc_l.append(icc)
    #     plt.scatter(tvc_lr, tvc_rl)
    #     plt.show()
    # quit()

    within_subj_icc_l = np.array(within_subj_icc_l)
    between_subj_icc_l = np.array(between_subj_icc_l)
    assert np.isnan(within_subj_icc_l).sum() == 0
    assert np.isnan(between_subj_icc_l).sum() == 0
    M_icc_within = np.nanmean(within_subj_icc_l)
    M_icc_between = np.nanmean(between_subj_icc_l)

    return M_icc_within, M_icc_between, icc_overall


def consistency_edge_rs_static(num_test=10_000, skip_other=True,
                               combine_regions=False, focus='combo'):
    sns = get_final_HCP_sns()[:1000]

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, combine_regions=combine_regions))

    p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all, p_no_all = (
        get_quads(skip_other=False, combine_regions=combine_regions))

    dd_w, dd_b, dd_o = get_network_connesitency(sns, p_d_ant, p_d_pos,
                                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                combine_regions=combine_regions)

    vv_w, vv_b, vv_o = get_network_connesitency(sns, p_v_ant, p_v_pos,
                                                p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                combine_regions=combine_regions)

    dv_ant_w, dv_ant_b, dv_ant_o = get_network_connesitency(sns, p_d_ant, p_v_ant,
                                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                            combine_regions=combine_regions)
    dv_pos_w, dv_pos_b, dv_pos_o = get_network_connesitency(sns, p_d_pos, p_v_pos,
                                                            p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                            combine_regions=combine_regions)

    print(f'{dd_w=:.3f}, {dd_b=:.3f}, {dd_o=:.3f}')
    print(f'{vv_w=:.3f}, {vv_b=:.3f}, {vv_o=:.3f}')
    print(f'{dv_ant_w=:.3f}, {dv_ant_b=:.3f}, {dv_ant_o=:.3f}')
    print(f'{dv_pos_w=:.3f}, {dv_pos_b=:.3f}, {dv_pos_o=:.3f}')

    # i, j = p_d_ant[0], p_d_pos[0]
    # get_rs_edge_tvc(sns, i, j,
    #                 p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
    #                 lr='LR', combine_regions=combine_regions)


def consistency_edge_rs_fluc(num_test=10_000, skip_other=True,
                             combine_regions=False, focus='combo'):
    sns = get_final_HCP_sns()[:250]

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, combine_regions=combine_regions))

    p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all, p_no_all = (
        get_quads(skip_other=False, combine_regions=combine_regions))

    ad_w, ad_b, ad_o = get_network_triangle_consistency(sns, p_d_ant, p_d_pos, p_v_ant,
                                                        p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                        combine_regions=combine_regions)
    print(f'{ad_w=:.3f}, {ad_b=:.3f}, {ad_o=:.3f}')

    pd_w, pd_b, pd_o = get_network_triangle_consistency(sns, p_d_pos, p_d_ant, p_v_pos,
                                                        p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                        combine_regions=combine_regions)
    print(f'{pd_w=:.3f}, {pd_b=:.3f}, {pd_o=:.3f}')

    av_w, av_b, av_o = get_network_triangle_consistency(sns, p_v_ant, p_v_pos, p_d_ant,
                                                        p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                        combine_regions=combine_regions)
    print(f'{av_w=:.3f}, {av_b=:.3f}, {av_o=:.3f}')

    pv_w, pv_b, pv_o = get_network_triangle_consistency(sns, p_v_pos, p_v_ant, p_d_pos,
                                                        p_d_ant_all, p_d_pos_all, p_v_ant_all, p_v_pos_all,
                                                        combine_regions=combine_regions)
    print(f'{pv_w=:.3f}, {pv_b=:.3f}, {pv_o=:.3f}')


if __name__ == '__main__':
    # consistency_edge_rs_static()
    consistency_edge_rs_fluc()
