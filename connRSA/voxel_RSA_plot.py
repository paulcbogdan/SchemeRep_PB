import numpy as np
from matplotlib import pyplot as plt

from atlas_utils import get_atlas
from connRSA.conn_regress import run_all_sn, get_title
from connRSA.conn_utils import get_BNA_ROIs
from old.plot_gen import my_plot_surf, get_split_cmap
from utils import pickle_wrap


def plot_basic():
    easy_override = False
    ISPC = False
    RSA = True
    semantic = False
    ERS_alt = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    regress_row = False
    stdize_by_run = True if trial_similarity == 'euc' else False

    ROI_foci = [f'{ROI}_BOLD' for ROI in get_BNA_ROIs()]

    ts_ROI, ts_BOLD, ts_conn = [], [], []
    kwargs_ = None
    kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
              'fp1': None, 'trial_similarity': trial_similarity,
              'second_order': second_order,
              'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,
              'regress_row': regress_row, 'four_tasks': four_tasks,
              }
    for ROI_focus in ROI_foci:
        kwargs_ = kwargs.copy()
        kwargs_['ROI_focus'] = ROI_focus
        kwargs_['ROIs_ctrl'] = []
        kw_outer = {'RSA': RSA, 'ISPC': ISPC, 'ERS_alt': ERS_alt,
                    'kwargs': kwargs_}
        t_ROIs_all, r_sqs_ROIs_all = pickle_wrap(run_all_sn, kwargs=kw_outer,
                                                 easy_override=easy_override)
        # t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
        #                                         easy_override=easy_override,
        #                                         )
        ts_ROI.append(t_ROIs_all)

    title, fn, fontsize = get_title(RSA, ISPC, kwargs_, 28)
    print(f'{ts_ROI=}')
    vmax = 5
    thresh = 2
    title_ROI = title.split(':')[0]# + ': ROIs'
    atlas = get_atlas()
    my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh)


def plot_just_cbar():
    cmap = get_split_cmap(5, 2, 'rainbow_r', blue_half=True)

    from matplotlib import rcParams, cm

    rcParams['axes.linewidth'] = 2  # set the value globally
    a = np.array([[0, 1]])
    plt.figure(figsize=(9, 2))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.05, 0.3, 0.9, 0.4])
    # norm = cm.colors.Normalize(vmax=5, vmin=0)
    cbar = plt.colorbar(orientation="horizontal", cax=cax,)
    cbar.ax.set_xticks([0.0, 0.4, 1.], ['0', '2', '5'], fontsize=40)
    cbar.ax.set_title('t-value', fontsize=40, pad=10)
    cbar.ax.xaxis.set_tick_params(width=2)
    # plt.tight_layout(rect=(0, 0.1, 1, 0.1))
    plt.show()

def plot_just_cbar2():
    cmap = get_split_cmap(4, 1, 'rainbow', blue_half=False)

    from matplotlib import rcParams, cm

    rcParams['axes.linewidth'] = 2  # set the value globally
    a = np.array([[0, 1]])
    plt.figure(figsize=(18, 3.5))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.15, 0.5, 0.7, 0.25])
    # norm = cm.colors.Normalize(vmax=5, vmin=0)
    cbar = plt.colorbar(orientation="horizontal", cax=cax,)
    cbar.ax.set_xticks([0, 0.375, 0.625, 1], ['-4\nLow PE', '-1', '1', '4\nHigh PE'],
                       fontsize=47)
    cbar.ax.set_title('t-value', fontsize=47, pad=15)
    cbar.ax.xaxis.set_tick_params(width=2)
    # plt.tight_layout(rect=(0, 0.1, 1, 0.1))
    plt.show()

if __name__ == '__main__':
    plot_just_cbar2()
