import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import pickle

import matplotlib.pyplot as plt
import scipy.stats as stats

def get_np_t_task(wl=False, schaefer=False, combine_regions=False, just_lr='L'):
    if wl:
        if schaefer:
            raise ValueError
        else:
            fp_rs = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_rs_wl.pkl'
            fp_task = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_task_wl.pkl'
    else:
        if schaefer:
            if just_lr is not None:
                lr_str = f'_{just_lr}'
                fp_rs = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(500, 1000)_rs_combo_schaefer{lr_str}.pkl'
                fp_task = fr'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(500, 1000)_task_combo_schaefer{lr_str}.pkl'
            elif combine_regions:
                fp_rs = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(36, 1000)_rs_combo_schaefer.pkl'
                fp_task = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(36, 1000)_task_combo_schaefer.pkl'
            else:
                # fp_rs = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(290, 1000)_rs_combo_schaefer.pkl'
                # fp_task = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(290, 1000)_task_combo_schaefer.pkl'
                fp_rs0 = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(1300, 500)_rs_combo_schaefer_h0.pkl'
                fp_rs1 = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(1300, 500)_rs_combo_schaefer_h1.pkl'
                fp_task0 = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(1300, 500)_task_combo_schaefer_h0.pkl'
                fp_task1 = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task\HCP_rs_x_task_corr__(1300, 500)_task_combo_schaefer_h1.pkl'
        else:
            fp_rs = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_rs.pkl'
            fp_task = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_task.pkl'
            # fp_rs = r'cache/HCP_rs_x_task_corr__(3432, 1000)_rs.pkl'
            # fp_task = r'cache/HCP_rs_x_task_corr__(3432, 1000)_task.pkl'

            # fp = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task'

    if schaefer and not combine_regions and just_lr is None:
        np_rs0 = pickle.load(open(fp_rs0, 'rb'))
        np_rs0 = np.array(np_rs0)
        np_rs1 = pickle.load(open(fp_rs1, 'rb'))
        np_rs1 = np.array(np_rs1)
        np_rs = np.concatenate([np_rs0, np_rs1], axis=1)
        np_rs = np_rs[:999, :]

        np_t0 = pickle.load(open(fp_task0, 'rb'))
        np_t0 = np.array(np_t0)
        np_t1 = pickle.load(open(fp_task1, 'rb'))
        np_t1 = np.array(np_t1)
        np_t = np.concatenate([np_t0, np_t1], axis=1)
        np_t = np_t[:999, :]

        print(f'Schaefer: {np_rs.shape=}')
    else:
        np_rs = pickle.load(open(fp_rs, 'rb'))
        np_rs = np.array(np_rs)
        np_t = pickle.load(open(fp_task, 'rb'))
        np_t = np.array(np_t)

    return np_rs, np_t

if __name__ == '__main__':
    #
    # # files generated via analyze_Study2B.py
    # # fp_rs = r'cache/HCP_rs_x_task_corr__(3432, 1000)_rs.pkl'
    # fp_rs = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(290, 1000)_rs_combo_schaefer.pkl'
    # # fp_rs = r'cache/rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_rs_wl.pkl'
    # np_rs = pickle.load(open(fp_rs, 'rb'))
    # np_rs = np.array(np_rs)
    # # fp_t = r'cache/HCP_rs_x_task_corr__(3432, 1000)_task.pkl'
    # fp_t = r'C:\PycharmProjects\SchemeRep\cache\rs_x_task/HCP_rs_x_task_corr__(290, 1000)_task_combo_schaefer.pkl'
    # # fp_t = r'cac he/rs_x_task/HCP_rs_x_task_corr__(3432, 1000)_task_wl.pkl'
    #
    # np_t = pickle.load(open(fp_t, 'rb'))
    # np_t = np.array(np_t)
    #
    # M_t = np.mean(np_t, axis=0)
    np_rs, np_t = get_np_t_task(wl=True, schaefer=False, combine_regions=False)
    M_t = np.mean(np_t, axis=0)
    M_rs = np.mean(np_rs, axis=0)
    n_roi = np_rs.shape[0]
    r, p = stats.spearmanr(M_t, M_rs)
    print(f'Mean corr: {r=:.2f}, {p=:.3f}')
    # print(np_t.shape)
    # quit()

    idxs = np.arange(n_roi)
    np.random.shuffle(idxs)
    np_rs = np_rs[idxs]
    np_t = np_t[idxs]

    # print(f'Across all ROIs: r = {r:.2f}, {p=:.4f}')
    # quit()

    across_l = []
    for i in range(n_roi):
        r, _ = stats.spearmanr(np_t[i], np_rs[i])
        across_l.append(r)

    within_l = []
    for j in range(np_t.shape[1]):
        r, _ = stats.spearmanr(np_t[:, j], np_rs[:, j])
        within_l.append(r)

    z = np.arctanh(across_l)

    t, p = stats.ttest_1samp(z, 0)
    N = len(z)
    print(f't[{N - 1}] = {t:.2f}, {p=:.4f}')
    t_within, p_within = stats.ttest_1samp(within_l, 0)
    N_within = len(within_l)
    print(f't_within[{N_within - 1}] = {t_within:.2f}, {p_within=:.4f}')

    plt.figure(figsize=(4.4, 2.5))
    plt.rcParams.update({'font.size': 14,
                         'font.sans-serif': 'Arial'})
    plt.gca().spines[['top', 'right']].set_visible(False)
    b = plt.hist(z, range=(-.1, .1), bins=40,
                 color='mediumorchid')[0]
    plt.plot([0, 0], [0, np.max(b)], 'k--', linewidth=1)
    plt.xlabel('Correlation (r)')
    plt.ylabel('Frequency\n(number of ROI sets)')
    M = np.nanmean(z)
    p_above_0 = np.mean(z > 0)
    plt.tight_layout()
    fp_out = r'result_pics/Fig5/Fig5B_histogram.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()
