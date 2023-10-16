from pickle_wrap import pickle_wrap

import utils
from analyze_ROIs import analyze_ROIs
from fMRI_proc import mass_RDM_x_RDM
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool
import pickle
from tqdm import tqdm
from functools import partial

class Timer:
    def __init__(self):
        self.t0 = time()

    def lap(self):
        elapsed = time() - self.t0
        self.t0 = time()
        return elapsed

def run_one_shuffle(fp_out, semantic=False, DNN_layer=2, combine_regions=False,
                    fp_fMRI_col='obj_fMRI', nsims=200):
    t = Timer()
    print(f'Starting: {fp_out=}')
    d = mass_RDM_x_RDM(age=1, cin=None,
                               DNN_layer=DNN_layer,
                               semantic=semantic,
                               bilateral=False,
                               combine_regions=combine_regions,
                               org_by_region=False,
                               fp_fMRI_col=fp_fMRI_col,
                               PCA_obj=True,
                               shuffle=True
                               )
    with open(fp_out, 'wb') as f:
        pickle.dump(d, f)
    print(f'Finished in {t.lap():.2f} seconds')

def prep_first_level_results(semantic=False, DNN_layer=2, combine_regions=False,
                             fp_fMRI_col='obj_fMRI', nsims=1000):
    fps_out = []
    for i in range(nsims):
        RSA_fn = utils.get_RSA_fn(None, 1, semantic, DNN_layer,
                                  fp_fMRI_col, PCA_obj=True,
                                  combine_regions=combine_regions,
                                  bilateral=False, vec_prod=False,
                                  org_by_region=False)
        RSA_fn = RSA_fn.replace('.pkl', '')
        dir_out = fr'cache/perms/{RSA_fn}'
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        fp_out = dir_out / Path(f'perm{i}.pkl')
        if fp_out.exists():
            continue
        fps_out.append(fp_out)

    with Pool(processes=4) as pool:
        f = partial(run_one_shuffle, semantic=semantic,
                    combine_regions=combine_regions, DNN_layer=DNN_layer,
                    fp_fMRI_col=fp_fMRI_col)
        pool.map(f, fps_out)


def get_perm_fps(semantic=False, DNN_layer=2, combine_regions=True,
                 fp_fMRI_col='obj_fMRI'):
    # Running a single one: 55 seconds each
    # Running in 2x parallel: 58 seconds each
    # Running in 7x parallel: 200 seconds each

    RSA_fn = utils.get_RSA_fn(None, 1, semantic, DNN_layer,
                              fp_fMRI_col, PCA_obj=True,
                              combine_regions=combine_regions,
                              bilateral=False, vec_prod=False,
                              org_by_region=False)
    RSA_fn = RSA_fn.replace('.pkl', '')
    dir_in = Path(fr'cache/perms/{RSA_fn}')
    return dir_in.glob('perm*.pkl')

def printout_ts_stats(ts_all):
    total_var = np.var(ts_all)
    within_var = np.mean(np.var(ts_all, axis=1))
    between_var = np.var(np.mean(ts_all, axis=1))
    r2 = between_var / (total_var)
    m = np.mean(ts_all)
    print(f'Mean: {m:.3f}')
    print(f'total_var: {total_var:.3f}, '
          f'within_var: {within_var:.3f}, '
          f'between_var: {between_var:.3f} [{r2:.1%}]')
    return m, within_var, between_var

def sim_normal(nsims=10000, N=34, ROIs=45):
    ts_all = []
    print(f'{ROIs=}')
    for i in range(nsims):
        ts = np.random.standard_t(N - 1, size=ROIs)
        ts_all.append(ts)
    ts_all = np.array(ts_all)
    print(ts_all.shape)
    print('Simulated')
    printout_ts_stats(ts_all)
    ts_max = np.max(ts_all, axis=1)
    ts_max = np.sort(ts_max)
    tiles = np.linspace(0, 1, nsims)
    return tiles, ts_max

def sim_multilevel(m, within_var, between_var, ROIs=45, nsims=10000):
    ts_all = []
    for i in range(nsims):
        t_m = np.random.normal(m, np.sqrt(between_var), size=1)
        ts = np.random.normal(t_m, np.sqrt(within_var), size=ROIs)
        ts_all.append(ts)
    ts_all = np.array(ts_all)
    ts_max = np.max(ts_all, axis=1)
    ts_max = np.sort(ts_max)
    tiles = np.linspace(0, 1, nsims)
    return tiles, ts_max

def run_statistics(semantic=False, DNN_layer=2, combine_regions=False,
                   fp_fMRI_col='obj_fMRI'):
    fps = get_perm_fps(semantic=semantic, DNN_layer=DNN_layer,
                       combine_regions=combine_regions, fp_fMRI_col=fp_fMRI_col,
                       )
    plt.rcParams.update({'font.size': 14})
    ts_all = []
    N = None
    nsims = 0
    for fp in tqdm(fps, desc='Getting t-values'):
        ts, N = analyze_ROIs(semantic=semantic, DNN_layer=DNN_layer,
                             combine_regions=combine_regions,
                             fp_fMRI_col=fp_fMRI_col, fp=fp, verbose=False,
                             req_all_N=True, key='obj')
        ts_all.append(ts)
        ts, N = analyze_ROIs(semantic=semantic, DNN_layer=DNN_layer,
                             combine_regions=combine_regions,
                             fp_fMRI_col=fp_fMRI_col, fp=fp, verbose=False,
                             req_all_N=True, key='scn')
        ts_all.append(ts)
        nsims += 1

    ts_all = np.array(ts_all)
    N_ROIs = ts_all.shape[1]
    ts_max = np.max(ts_all, axis=1)
    ts_max = np.sort(ts_max)
    tiles = np.linspace(0, 1, len(ts_max))
    m, within_var, between_var = printout_ts_stats(ts_all)
    plt.plot([0.95, 0.95], [0, 5], 'k--')
    bonf = 0.05 / N_ROIs
    t_if_ind = -stats.t.ppf(bonf, N-1)

    plt.plot([0.8, 1], [t_if_ind, t_if_ind], 'k--')
    plt.plot(tiles, ts_max, 'r', label=f'Shuffled real data ($n_{{sims}} = '
                                       f'${nsims*2})',
             linewidth=2, zorder=10)
    plt.plot(*sim_normal(N=N, ROIs=N_ROIs), 'g', label=f'Simulated parametric-t'
                                                       f' ($df=${N-1})')
    z_label = f'\n(m={m:.2f}, $\sigma^2$={within_var:.2f}, ' \
              f'$\\tau^2_{{00}}$={between_var:.2f})'
    plt.plot(*sim_multilevel(m, within_var, between_var, ROIs=N_ROIs), 'b',
             label='Simulated multilevel-z' + z_label)
    plt.title(f'# subjects = {N}, # ROIs = {N_ROIs}')
    plt.ylabel('Max t-value')
    plt.xlabel('Percentile')
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


if __name__ == '__main__':
    # prep_first_level_results()
    run_statistics()