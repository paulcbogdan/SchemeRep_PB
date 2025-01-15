import itertools

import numpy as np
from scipy import stats
from tqdm import tqdm

from Study1B.analyze_plot_Fig3 import get_study1b_ar, get_sn_roi_ar
from Study2B.analyze_Study2B import get_final_HCP_sns, get_quads
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from marinate.pkld import pkld
from old.plot_gen import my_plot_surf


@pkld
def get_study1b_conn_mat(sn, combine_regions=True):
    kw = {'sn': sn, 'lr': 'lr', 'combine_regions': combine_regions,
          'bilateral': False, 'reg_global': False,
          'no_compcor': False, 'rs': True}
    ar_lr = pickle_wrap(get_sn_roi_ar, kwargs=kw, verbose=-1)
    corr_lr = np.corrcoef(ar_lr)
    kw['lr'] = 'rl'
    ar_rl = pickle_wrap(get_sn_roi_ar, kwargs=kw, verbose=-1)
    # # ar_lr = get_sn_roi_ar(sn, 'lr', combine_regions=combine_regions,
    # #                    bilateral=False, reg_global=False,
    # #                    no_compcor=False, rs=True)
    # corr_lr = np.corrcoef(ar_lr)
    # ar_rl = get_sn_roi_ar(sn, 'rl', combine_regions=combine_regions,
    #                       bilateral=False, reg_global=False,
    #                       no_compcor=False, rs=True)
    corr_rl = np.corrcoef(ar_rl)
    corr = (corr_lr + corr_rl) / 2
    return corr


@pkld(overwrite=True)
def get_BOLD_ef(sn, combine_regions=True, bilateral=False,
                only='loss', learning_rate=0.3,
                drop_first=False, reset_trial0=True,
                ):
    if only == 'wl':
        return get_BOLD_loss_ef(sn, combine_regions=combine_regions,
                                bilateral=bilateral, learning_rate=learning_rate,
                                drop_first=drop_first, reset_trial0=reset_trial0)
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'only': only, 'learning_rate': learning_rate,
          'drop_first': drop_first, 'reset_trial0': reset_trial0}
    if only == 'combo':
        kw['only'] = 'loss'
        ef_loss = get_BOLD_ef(sn, **kw)
        kw['only'] = 'win'
        ef_win = get_BOLD_ef(sn, **kw)
        return (ef_loss + ef_win) / 2

    ar_high_lr, ar_low_lr = get_study1b_ar(sn, rl_lr='lr', **kw)
    ar_high_rl, ar_low_rl = get_study1b_ar(sn, rl_lr='rl', **kw)
    # print(ar_high_lr)
    # quit()

    M_high_lr = np.nanmean(ar_high_lr, axis=1)
    M_low_lr = np.nanmean(ar_low_lr, axis=1)
    M_high_rl = np.nanmean(ar_high_rl, axis=1)
    M_low_rl = np.nanmean(ar_low_rl, axis=1)
    ef = (M_high_lr - M_low_lr) + (M_high_rl - M_low_rl)
    return ef


@pkld
def get_BOLD_loss_ef(sn, combine_regions=True, bilateral=False,
                     learning_rate=0.3,
                     drop_first=False, reset_trial0=True, ):
    kw = {'combine_regions': combine_regions, 'bilateral': bilateral,
          'only': 'loss', 'learning_rate': learning_rate,
          'drop_first': drop_first, 'reset_trial0': reset_trial0}
    ar_high_lr, ar_low_lr = get_study1b_ar(sn, rl_lr='lr', **kw)
    ar_high_lr, ar_low_lr = np.nanmean(ar_high_lr, axis=1), np.nanmean(ar_low_lr, axis=1)
    ar_high_rl, ar_low_rl = get_study1b_ar(sn, rl_lr='rl', **kw)
    ar_high_rl, ar_low_rl = np.nanmean(ar_high_rl, axis=1), np.nanmean(ar_low_rl, axis=1)
    ar_loss = (ar_high_lr + ar_low_lr) + (ar_high_rl + ar_low_rl)

    kw['only'] = 'win'
    ar_high_lr, ar_low_lr = get_study1b_ar(sn, rl_lr='lr', **kw)
    ar_high_lr, ar_low_lr = np.nanmean(ar_high_lr, axis=1), np.nanmean(ar_low_lr, axis=1)
    ar_high_rl, ar_low_rl = get_study1b_ar(sn, rl_lr='rl', **kw)
    ar_high_rl, ar_low_rl = np.nanmean(ar_high_rl, axis=1), np.nanmean(ar_low_rl, axis=1)
    ar_win = (ar_high_lr + ar_low_lr) + (ar_high_rl + ar_low_rl)
    return ar_loss - ar_win


def prep_quad_ROIs(skip_other=True, combine_regions=False,
                   num_test=10_000, random=False):
    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, combine_regions=combine_regions))
    if random:
        idxs = np.arange(54 if combine_regions else 246)[::2]
        np.random.shuffle(idxs)
        p_d_ant = idxs[:len(p_d_ant)]
        idxs = idxs[len(p_d_ant):]
        p_d_pos = idxs[:len(p_d_pos)]
        idxs = idxs[len(p_d_pos):]
        p_v_ant = idxs[:len(p_v_ant)]
        idxs = idxs[len(p_v_ant):]
        p_v_pos = idxs[:len(p_v_pos)]

    np.random.seed(0)
    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    combos = list(combos)
    np.random.shuffle(combos)

    combos_ = []
    for (a, b, c, d) in combos:
        if len(combos_) >= num_test:
            continue
        if len({a, b, c, d}) < 4:
            continue
        combos_.append((a, b, c, d))
    combos = combos_
    return combos


def study1b_BOLD_x_RS(combine_regions=False, only='win',
                      skip_other=True):
    sns = get_final_HCP_sns()
    # sns = sns[:750]
    # sns = sns[:5]
    # sns = sns[::-1]
    np.random.seed(1)
    combos = prep_quad_ROIs(random=True, skip_other=skip_other,
                            combine_regions=combine_regions,
                            num_test=10_000)

    ROI_ROIs_all = []
    ROI_efs_all = []
    sn_rs_all = []
    for sn in tqdm(sns, desc='load sn BOLD_x_rs'):
        ROI_ROI_rs = get_study1b_conn_mat(sn, combine_regions=combine_regions)
        ROI_efs = get_BOLD_ef(sn, combine_regions=combine_regions,
                              only=only, drop_first=True)
        ROI_ROIs_all.append(ROI_ROI_rs)
        ROI_efs_all.append(ROI_efs)

        efs = []
        conns = []
        for da, dp, va, vp in combos:
            da = [da, da + 1]
            dp = [dp, dp + 1]
            va = [va, va + 1]
            vp = [vp, vp + 1]
            total_ef = ROI_efs[va] + ROI_efs[vp] - ROI_efs[da] - ROI_efs[dp]
            if len(total_ef.shape):
                total_ef = np.nanmean(total_ef)
            ROI_ROI = -(ROI_ROI_rs[va, da] + ROI_ROI_rs[vp, dp] +
                        ROI_ROI_rs[va, dp] + ROI_ROI_rs[da, vp]) / 4
            if len(ROI_ROI.shape):
                ROI_ROI = np.nanmean(ROI_ROI)
            # ROI_ROI += (ROI_ROI_rs[va, vp] + ROI_ROI_rs[da, dp]) / 2
            # bigger ef -> more negative conn
            efs.append(total_ef)
            conns.append(ROI_ROI)
        r, p = stats.spearmanr(efs, conns, nan_policy='omit')
        # print(f'{sn} | {r=:.2f}, {p=:.4f}')
        sn_rs_all.append(r)
        if len(sn_rs_all) % 10 == 0:
            M = np.nanmean(sn_rs_all)
            SE = np.nanstd(sn_rs_all) / len(sn_rs_all) ** .5
            t, p = stats.ttest_1samp(sn_rs_all, 0)
            print(f'{M=:.2f} ({SE=:.2f}) | t[{len(sn_rs_all) - 1}] = {t:.2f}, {p:.4f}')
            # ROI_ROI_low_PE = ROI_ROI_rs[va, vp] + ROI_ROI_rs[da, dp]
            # ROI_ROI_high_PE = ROI_ROI_rs[va, vp] + ROI_ROI_rs[da, dp]
            # ROI_ROI_ef = ROI_ROI_high_PE - ROI_ROI_low_PE

    ROI_efs_all = np.array(ROI_efs_all)
    ROI_ts = (np.nanmean(ROI_efs_all, axis=0) /
              np.nanstd(ROI_efs_all, axis=0) *
              np.sqrt(ROI_efs_all.shape[0]))


def plot_study1b_BOLD(combine_regions=False, only='wl'):
    sns = get_final_HCP_sns()
    # sns = sns[:750]
    # sns = sns[:5]
    # sns = sns[::-1]
    ROI_efs_all = []
    for sn in tqdm(sns, desc='plot_study1b_BOLD'):
        # if only == 'wl':
        #     ROI_efs = get_BOLD_loss_ef(sn, combine_regions=combine_regions,
        #                                drop_first=True
        #                                )
        # else:
        ROI_efs = get_BOLD_ef(sn, combine_regions=combine_regions,
                              only=only, drop_first=True
                              )

        ROI_efs_all.append(ROI_efs)
    ROI_efs_all = np.array(ROI_efs_all)
    ROI_ts = (np.nanmean(ROI_efs_all, axis=0) /
              np.nanstd(ROI_efs_all, axis=0) *
              np.sqrt(ROI_efs_all.shape[0]))

    atlas = get_atlas(combine_regions=combine_regions)

    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=False, combine_regions=combine_regions))

    for l, name in zip([p_d_ant, p_d_pos, p_v_ant, p_v_pos],
                       ['PFC', 'Parietal', 'ATL', 'Occipital']):
        efs = ROI_efs_all[:, l]
        efs = np.nanmean(efs, axis=1)
        t, p = stats.ttest_1samp(efs, 0)
        print(f'{name} | {t=:.2f}, {p=:.4f}')
    # quit()

    my_plot_surf(ROI_ts, atlas, f'Study1B BOLD: {only}', vmax=7, thresh=3,
                 only_positive=False, cmap='turbo', )


if __name__ == '__main__':
    study1b_BOLD_x_RS()
    # plot_study1b_BOLD()
    # get_BOLD_ef('100206')
