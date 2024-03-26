import numpy as np
from matplotlib import pyplot as plt


def plot_hz_corrs(name2r):
    def get_range(hz):

        if hz < 4:
            return cm(0.05)
        elif hz < 9:
            return cm(0.3)
        elif hz < 14:
            return cm(0.63)
        elif hz < 30:
            return cm(0.75)
        else:
            return cm(0.95)

    cm = plt.get_cmap('Spectral_r')
    # print(name2r)
    # quit()
    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(6, 4))

    for hz in range(1, 51):
        name = f'r{hz}'
        SE = np.nanstd(name2r[name]) / np.sqrt(len(name2r[name]))
        M = np.nanmean(name2r[name])
        high = M + 1 * SE
        low = M - 1 * SE
        c = get_range(hz)
        plt.plot([hz-0.05, hz-0.05], [low, high], color='gray', alpha=1,
                 linewidth=0.75)
        plt.plot([hz+0.05, hz+0.05], [low, high], color=c, alpha=1,
                 linewidth=0.75)

        plt.plot([hz, hz], [low, high], color=c, alpha=0.25,
                 linewidth=3.0)
        plt.plot([hz-.2, hz+.2], [high, high], color='gray')
        plt.plot([hz-.2, hz+.2], [low, low], color='gray')
        plt.plot(hz, M, 'o', color=c, markersize=3)


    plt.xlabel('Hz', labelpad=7)
    plt.ylabel(r'Connectivity (d) x Power (A$^{\rm 2}$)',# + '\nCorrelation',
               labelpad=7)

    drag = .015
    plt.ylim(0, 0.089 - drag)
    plt.xlim(0.5, 50.5)
    plt.yticks([0, 0.02, 0.04, 0.06, 0.08])
    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    # plt.gca().set_facecolor('whitesmoke')
    plt.text(1, 0.08 - drag, 'Delta', fontsize=12, c=get_range(2.5),
             ha='left')
    plt.text(6, 0.087 - drag, 'Theta', fontsize=12, c='green',
             ha='center')
    plt.text(11.5, 0.08 - drag, 'Alpha', fontsize=12, c='goldenrod',
             ha='center')
    plt.text(22, 0.08 - drag, 'Beta', fontsize=12, c='orange',
             ha='center')
    plt.text(40, 0.08 - drag, 'Gamma', fontsize=12, c='red',
             ha='center')

    # pc = patches.Rectangle((0.25, 0), 3, 0.075,
    #                        color=get_range(2.5), alpha=0.1)
    # plt.gca().add_patch(pc)
    #
    # pc = patches.Rectangle((3.25, 0), 4, 0.075,
    #                        color='green', alpha=0.1)
    # plt.gca().add_patch(pc)

    plt.gca().spines[['right', 'top',]].set_visible(False)
    plt.tight_layout()
    plt.savefig('result_pics/other/hz_corrs.png', dpi=300)
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
