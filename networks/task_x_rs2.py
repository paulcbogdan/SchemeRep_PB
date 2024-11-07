import itertools

import matplotlib.pyplot as plt
from tqdm import tqdm

from group_anat_fluc import get_group_avg_rs_r, get_group_level_d
from Study2B.analyze_Study2B import get_quads
from scipy import stats
import numpy as np
import statistics as stat
from random import shuffle, random

class Colors:
    '''
    Codes for printing colored text to the terminal. Helps things look nice
    Taken from: https://stackoverflow.com/questions/37340049/how-do-i-print-colored-output-to-the-terminal-in-python
    '''
    RED = '\033[31m'
    ENDC = '\033[m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

def print_list_stats(l: list) -> None:
    '''
    Prints some basic stats about a list of numbers, including the count, mean,
        median, min, max, and p(under 50%) and p(above 50%) of the list.
        Additionally, it plots details on the percentiles of values in the list.
    This is useful for getting a quick summary of the permute-testing
        results or data on the different component classifiers.

    :param l: list of numbers
    '''
    if len(l) < 2:
        print(f'List has fewer than two elements: {l}.')
        return
    elif all(x == l[0] for x in l):
        print(f'All values are the same, {len(l)=}.')
        return
    l.sort()
    print(f'Number of items: {len(l)}')
    SD_l = stat.stdev(l)
    SD2 = stat.mean(l) + stat.stdev(l) * 2
    print(f'Mean item: {stat.mean(l):.3f} [{SD_l:.3f}], high = {SD2:.3f}')
    print(f'Median item: {stat.median(l):.3f}')
    print(f'Min item: {min(l):.3f}')
    print(f'Max item: {max(l):.3f}')
    p_above = stat.mean([int(i > .50001) for i in l])
    p_below = stat.mean([int(i < .49999) for i in l])
    print(f'p(under 50%): {p_below:.3f} | p(above 50%): {p_above:.3f}')
    p_str = 'Percentile: Accuracy | '
    for p_cutoff in [1.0, .75, .5, .25, .1, .05, .01, .005, .001]:
        idx = min(int(len(l) * (1 - p_cutoff) + .999), len(l) - 1)
        p_str += f'{p_cutoff}: {Colors.BLUE}{l[idx]:.4f}{Colors.ENDC}, '
    print(p_str[:-2])  # the ':-2' crops out the comma and space at the end
    print()

def do_group_task_x_rs(combine_regions=True,
                       n='8', std_d=True, shuffle_seed=None, ix=True,
                       num_test=1_000, skip_other=False, sub_mean=False,
                       ctrl=False, sub_subj_mean=True):
    p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no = (
        get_quads(skip_other=skip_other, all_roi=False, anat_ver=3,
                  combine_regions=combine_regions, ))

    p_d_ant = tuple(p_d_ant)
    p_d_pos = tuple(p_d_pos)
    p_v_ant = tuple(p_v_ant)
    p_v_pos = tuple(p_v_pos)
    p_no = tuple(p_no)

    combos = itertools.product(p_d_ant, p_d_pos, p_v_ant, p_v_pos)
    combos = list(combos)
    shuffle(combos)
    num_pos = len(p_d_ant) * len(p_d_pos) * len(p_v_ant) * len(p_v_pos)
    print(f'{num_pos=}')
    # og_r = get_group_avg_rs_r(p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no,
    #                           ix=ix, combine_regions=combine_regions, n=n,
    #                           sn_vals=True, ctrl=False)
    # og_d = get_group_level_d(p_d_ant, p_d_pos, p_v_ant, p_v_pos, p_no,
    #                          ix=ix, combine_regions=combine_regions,
    #                          n=n, std_d=std_d, shuffle_seed=shuffle_seed,
    #                          sn_vals=True, ix_lat=False)
    # og_d[og_d < -.4] = -.4
    # plt.scatter(og_r, og_d)
    # plt.show()
    # r, p = stats.spearmanr(og_r, og_d, nan_policy='omit')
    # print(f'OG: {r=:.3f}, {p=:.3f}')
    # quit()

    combos_ = []
    for (a, b, c, d) in combos:
        if len(combos_) >= num_test:
            continue

        # if random() > num_test / num_pos:
        #     continue
        if len({a, b, c, d}) < 4:
            continue
        combos_.append((a, b, c, d))
    combos = combos_

    rs = []
    og_rs = []
    og_ds = []
    for (a, b, c, d) in tqdm(combos):

        if skip_other:
            pda_i = [a, a + 1]
            pdp_i = [b, b + 1]
            pva_i = [c, c + 1]
            pvp_i = [d, d + 1]
        else:
            pda_i = [a]
            pdp_i = [b]
            pva_i = [c]
            pvp_i = [d]

        pda_i = tuple(pda_i)
        pdp_i = tuple(pdp_i)
        pva_i = tuple(pva_i)
        pvp_i = tuple(pvp_i)

        og_r = get_group_avg_rs_r(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                  ix=ix, combine_regions=combine_regions, n=n,
                                  sn_vals=True, ctrl=ctrl)
        og_d = get_group_level_d(pda_i, pdp_i, pva_i, pvp_i, p_no,
                                 ix=ix, combine_regions=combine_regions,
                                 n=n, std_d=std_d, shuffle_seed=shuffle_seed,
                                 sn_vals=True)
        if sub_subj_mean:
            og_r -= np.nanmean(og_r)
            og_d -= np.nanmean(og_d)
        og_rs.append(og_r)
        og_ds.append(og_d)

    og_rs = np.array(og_rs)
    og_ds = np.array(og_ds)
    if sub_mean:
        og_rs -= np.nanmean(og_rs, axis=0)
        og_ds -= np.nanmean(og_ds, axis=0)
    for og_r, og_d in zip(og_rs, og_ds):
        r, p = stats.spearmanr(og_r, og_d)
        rs.append(r)
    r = np.nanmean(rs)
    M_r = np.nanmean(rs)
    t = np.nanmean(rs) / np.nanstd(rs) * np.sqrt(len(rs))
    if shuffle_seed is None:
        print(f'{t=:.3f}, {M_r=:.3f}')
    # print(rs)

    # assert np.sum(np.isnan(og_r)) == 0 and np.sum(np.isnan(og_d)) == 0
    # n_sn = len(og_r)
    # plt.title(f'{r=:.3f}, {p=:.3f}, N = {n_sn}')
    # plt.scatter(og_r, og_d)
    # plt.show()

    return t, M_r

def shuffle_text_r():
    rs = []
    M_rs = []
    for seed in range(100):
        t, M_r = do_group_task_x_rs(shuffle_seed=seed)
        rs.append(t)
        M_rs.append(M_r)
        print(f'Shuffle: {t=:.2f}, {M_r=:.3f}')
        if seed % 5 == 4:
            print_list_stats(rs)
            print_list_stats(M_rs)


if __name__ == '__main__':
    do_group_task_x_rs()
    # shuffle_text_r()