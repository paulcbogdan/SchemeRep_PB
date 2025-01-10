import os
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import pickle

import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == '__main__':

    # files generated via analyze_Study2B.py
    fp_rs = r'cache/HCP_rs_x_task_corr__(3432, 1000)_rs.pkl'
    np_rs = pickle.load(open(fp_rs, 'rb'))
    np_rs = np.array(np_rs)
    fp_t = r'cache/HCP_rs_x_task_corr__(3432, 1000)_task.pkl'
    np_t = pickle.load(open(fp_t, 'rb'))
    np_t = np.array(np_t)
    n_roi = np_rs.shape[0]

    M_t = np.mean(np_t, axis=0)
    M_rs = np.mean(np_rs, axis=0)
    r, p = stats.spearmanr(M_t, M_rs)
    print(f'Across all ROIs: r = {r:.2f}, {p=:.4f}')
    quit()

    across_l = []
    for i in range(n_roi):
        r, _ = stats.spearmanr(np_t[i], np_rs[i])
        across_l.append(r)
    # for j in range(np_t.shape[1]):
    #     r, _ = stats.spearmanr(np_t[:, j], np_rs[:, j])
    #     across_l.append(r)


    z = np.arctanh(across_l)

    t, p = stats.ttest_1samp(z, 0)
    N = len(z)
    print(f't[{N - 1}] = {t:.2f}, {p=:.4f}')

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
