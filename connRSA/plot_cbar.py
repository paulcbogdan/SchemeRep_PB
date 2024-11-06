import numpy as np
from matplotlib import pyplot as plt

from atlas_utils import get_atlas, get_BNA_ROIs
from connRSA.conn_regress import run_all_sn, get_title
from old.plot_gen import my_plot_surf, get_split_cmap
from utils import pickle_wrap



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

def plot_just_cbar2(setting='conn_RSA_dif'):
    # cmap = get_split_cmap(4, 1, 'turbo', blue_half=False)
    # cmap = get_split_cmap(0.3, 0, 'turbo', blue_half=False)

    if setting == 'conn_RSA_inferno':
        cmap = get_split_cmap(8, 2.4, 'inferno',
                              blue_half=False, inferno_half=True)
    elif setting == 'conn_RSA_dif':
        cmap = get_split_cmap(4, 2.4, 'turbo', blue_half=False,
                              full_range=True)
    elif setting == 'conn_RSA_cortex':
        cmap = get_split_cmap(5, 2.5, 'rainbow_r', blue_half=True,
                              black_line=0.0002)
    else:
        cmap = get_split_cmap(0.3, 0, 'turbo', blue_half=False)

    # cmap = get_split_cmap(0.3, 0, 'RdYlBu', blue_half=False)
    cmap = get_split_cmap(0.3, 0, 'turbo', blue_half=False)


    from matplotlib import rcParams, cm

    # set default font to ARial
    rcParams['font.sans-serif'] = 'Arial'

    rcParams['axes.linewidth'] = 2  # set the value globally
    a = np.array([[0, 1]])
    plt.figure(figsize=(18, 4.25))# if setting == 'conn_RSA_dif' else 3.5))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.15, 0.5, 0.7, 0.2])
    if setting != 'conn_RSA_inferno' and setting != 'conn_RSA_dif':
        plt.title('t-value', fontsize=80, pad=30)
    # cax = plt.axes([0.15, 0.5, 0.7, 0.25])

    # norm = cm.colors.Normalize(vmax=5, vmin=0)
    cbar = plt.colorbar(orientation="horizontal", cax=cax,)
    # cbar.ax.set_xticks([0, 0.375, 0.625, 1],
    #                    ['-4\nLow PE', '-1', '1', '4\nHigh PE'],
    #                    fontsize=47)

    if setting == 'conn_RSA_inferno':
        cbar.ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '2', '4', '6', '8'],
                           fontsize=60, )
        cbar.ax.tick_params(axis='x', pad=15, length=10)
    elif setting == 'conn_RSA_dif':
        cbar.ax.set_xticks([0, 0.25, 0.5, 0.75, 1],
                           ['-4\n(Local)',
                            '-2', '0', '2',
                            '4\n(Distributed)'], fontsize=60,)
        cbar.ax.tick_params(axis='x', pad=15, length=10)
    elif setting == 'conn_RSA_cortex':
        cbar.ax.set_xticks([0, 0.5, 1], ['0', '2.5', '5'], fontsize=47)
    else:
        cbar.ax.set_xticks([0, 0.5, 1], ['-.3', '.0', '.3'], fontsize=47)
        cbar.outline.set_visible(False)

    # cbar.ax.set_title('t-value', fontsize=47, pad=15)
    cbar.ax.set_xticks([0, 0.5, 1], ['-.3', '.0', '.3'], fontsize=47)
    # cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_tick_params(width=2)
    # plt.tight_layout(rect=(0, 0.1, 1, 0.1))
    plt.show()

if __name__ == '__main__':
    # plot_just_cbar2()
    plot_just_cbar2('conn_RSA_dif')
