from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def plot_hz_corrs(name2r, effect_size=False, t_vals=False, plot_se=True):
    def get_range(hz):

        if hz < 4:
            return cm(0.05)
        elif hz < 8:
            return cm(0.3)
        elif hz < 13:
            return cm(0.63)
        elif hz < 30:
            return cm(0.75)
        else:
            return cm(0.95)

    cm = plt.get_cmap('Spectral_r')
    # print(name2r)
    # quit()
    plt.rcParams.update({'font.size': 14,
                         'font.sans-serif': 'Arial'})

    # plt.figure(figsize=(6, 4))
    plt.figure(figsize=(6, 3.35))


    upper = 101 if 'r100' in name2r else 51
    hz_all = []
    M_all = []
    for hz in range(1, upper):
        name = f'r{hz}'
        if 'r100' in name2r:
            if hz == 1:
                print('Skip 0.5 Hz')
                continue
            hz /= 2
        SD = np.nanstd(name2r[name], ddof=1)
        SE = SD / np.sqrt(len(name2r[name]))
        M = np.nanmean(name2r[name])

        if t_vals:
            M /= SE
            SE = 1
        elif effect_size:
            M /= SD

            alt_SE = np.sqrt((1 / len(name2r[name])) +
                             ((M ** 2) / (2*len(name2r[name]))))

            SE /= SD
            print(f'{alt_SE=:.5f}, {SE=:.5f}')
            print(f'{M=:.5f}, {SD=:.5f}')



        high = M + 1 * SE
        low = M - 1 * SE
        c = get_range(hz)
        if not t_vals and plot_se:
            plt.plot([hz-0.05, hz-0.05], [low, high], color='gray',
                     alpha=0.8,
                     linewidth=0.5)
            plt.plot([hz+0.05, hz+0.05], [low, high], color=c,
                     alpha=0.6,
                     linewidth=0.75)

            plt.plot([hz, hz], [low, high], color=c, alpha=0.25,
                     linewidth=3.0)
            plt.plot([hz-.2, hz+.2], [high, high], color='gray',
                     alpha=1, linewidth=0.5)
            plt.plot([hz-.2, hz+.2], [low, low], color='gray',
                     alpha=1, linewidth=0.5)
        plt.plot(hz, M, 'o', color=c, markersize=3)
        hz_all.append(hz)
        M_all.append(M)
    if t_vals:
        plt.plot(hz_all, M_all, color='k', alpha=0.5, linewidth=0.5,
                 zorder=-1)


    plt.xlabel('Hz', labelpad=7)
    if effect_size:
        plt.ylabel(r'Effect size', labelpad=7)
    else:
        plt.ylabel(r'Connectivity (d) x Power (A$^{\rm 2}$)',# + '\nCorrelation',
                   labelpad=7, fontsize=14)

    if t_vals:
        height = 6
        height_theta = 7.1
        drag = 0.7
        plt.yticks([0, 1, 2, 3, 4, 5, 6,])
        plt.ylim(0, 6.)
    elif effect_size:
        height = 1.47
        height_theta = 1.63
        drag = 0.087
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4,])
        plt.yticks([0, 0.3, 0.6, 0.9, 1.2, ])

        plt.ylim(0, 1.5)
    else:
        height = 0.068
        height_theta = 0.075
        drag = .007  # 0.015
        plt.ylim(0, 0.073 - drag)
        plt.yticks([0, 0.02, 0.04, 0.06])
    plt.xlim(0.0, 50.5)
    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    # plt.gca().set_facecolor('whitesmoke')
    plt.text(1, height - drag, 'Delta', fontsize=14, c=get_range(2.5),
             ha='left')
    plt.text(6, height_theta - drag, 'Theta', fontsize=14, c='green',
             ha='center')
    plt.text(11.5, height - drag, 'Alpha', fontsize=14, c='goldenrod',
             ha='center')
    plt.text(22, height - drag, 'Beta', fontsize=14, c='orange',
             ha='center')
    plt.text(40, height - drag, 'Gamma', fontsize=14, c='red',
             ha='center')

    # pc = patches.Rectangle((0.25, 0), 3, 0.075,
    #                        color=get_range(2.5), alpha=0.1)
    # plt.gca().add_patch(pc)
    #
    # pc = patches.Rectangle((3.25, 0), 4, 0.075,
    #                        color='green', alpha=0.1)
    # plt.gca().add_patch(pc)

    plt.gca().spines[['right', 'top',]].set_visible(False)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    ts = int(datetime.now().timestamp())
    if effect_size:
        fp_out = fr'result_pics/other/hz_effect_size_{ts}.png'
    else:
        fp_out = fr'result_pics/other/hz_corr_{ts}.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()

def plot_psd(PSDs):
    cm = plt.get_cmap('viridis')
    nfreq = np.array(PSDs).shape[1]
    colors = [cm(i / nfreq) for i in range(nfreq)]
    colors = np.array(colors)
    M_PSD = np.nanmean(np.array(PSDs), axis=0)
    for i, l in enumerate(M_PSD):
        plt.plot(np.arange(1, 52), l, color=colors[i],
                 alpha=.7)
    plt.xlim(1, 50)
    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.xlabel('Hz')
    plt.ylabel('Power (dB)')
    plt.gca().spines[['bottom', 'right', 'top']].set_visible(False)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# ts = int(datetime.now().timestamp())
# print(ts)