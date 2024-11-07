import numpy as np
from matplotlib import pyplot as plt


def plot_hz_corrs(name2r, effect_size=False, t_vals=False, plot_se=True,
                  flip_VD=True, high_words=False, just_frontal=False):
    def get_range(hz):
        if hz < 1:
            return 'dimgray'
        elif hz < 4:
            return cm(0.075)
        elif hz < 8:
            return cm(0.3)
        elif hz < 13:
            return cm(0.6)
        elif hz < 30:
            return cm(0.75)
        else:
            return cm(0.95)

    def get_range_hard(hz):
        if hz < 1:
            return 'dimgray'
        elif hz < 4:
            return cm(0.075)
        elif hz < 8:
            return 'green'
        elif hz < 13:
            return 'goldenrod'
        elif hz < 30:
            return cm(0.75)
        else:
            return cm(0.95)

    cm = plt.get_cmap('Spectral_r')

    plt.rcParams.update({'font.size': 14,
                         'font.sans-serif': 'Arial'})

    plt.figure(figsize=(6, 3.35))

    hz_all = []
    M_all = []
    if 'r100' in name2r or 'r100.0' in name2r:
        upper = 100
        cnt = 200
    else:
        upper = 50
        cnt = 100
    for hz in np.linspace(0.5, upper, cnt):
        name = f'r{hz:.1f}'

        SD = np.nanstd(name2r[name], ddof=1)
        SE = SD / np.sqrt(len(name2r[name]))
        M = np.nanmean(name2r[name])

        if t_vals:
            M /= SE
            SE = 1
        elif effect_size:
            M /= SD
            SE /= SD

        high = M + 1 * SE
        low = M - 1 * SE
        c = get_range(hz)
        c_h = get_range_hard(hz)
        if not t_vals and plot_se:
            plt.plot([hz - 0.05, hz - 0.05], [low, high], color='gray',
                     alpha=0.8,
                     linewidth=0.5)
            plt.plot([hz + 0.05, hz + 0.05], [low, high], color=c_h,
                     alpha=0.8,
                     linewidth=0.75)

            plt.plot([hz, hz], [low, high], color=c, alpha=0.3,
                     linewidth=2.5)
            plt.plot([hz - .2, hz + .2], [high, high], color='gray',
                     alpha=1, linewidth=0.5)
            plt.plot([hz - .2, hz + .2], [low, low], color='gray',
                     alpha=1, linewidth=0.5)
        plt.plot(hz, M, 'o', color=c_h, markersize=2.3)
        hz_all.append(hz)
        M_all.append(M)
    if t_vals:
        plt.plot(hz_all, M_all, color='k', alpha=0.5, linewidth=0.5,
                 zorder=-1)

    plt.xlabel('Hz', labelpad=7)
    if effect_size:
        plt.ylabel(r'Effect size', labelpad=7)
    else:
        if flip_VD:
            plt.ylabel(f'Connectivity (|PA-VD|)\n× ' + r'Power (A$^{\rm 2}$)',
                       labelpad=7, fontsize=13)
        else:
            plt.ylabel(f'Connectivity (|PA-VD|)\n× ' + r'Power (A$^{\rm 2}$)',
                       labelpad=7, fontsize=13)

    if t_vals:
        height = 6
        height_theta = 7.1
        drag = 0.7
        plt.yticks([0, 1, 2, 3, 4, 5, 6, ])
        plt.ylim(0, 6.)
    elif effect_size:
        height = 1.53
        height_theta = 1.68
        drag = 0.087
        plt.yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, ])

        plt.ylim(0, 1.65)
    else:
        height = 0.068
        height_theta = 0.075
        drag = .007  # 0.015
        plt.ylim(0, 0.073 - drag)
        plt.yticks([0, 0.02, 0.04, 0.06])
    plt.xlim(0.0, 50.5)
    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    if high_words:
        plt.text(3.25, height - drag, 'Delta', fontsize=14, c=get_range(2.5),
                 ha='center')
        plt.text(5.75, height_theta - drag, 'Theta', fontsize=14, c='green',
                 ha='center')
        plt.text(9.75, height - drag, 'Alpha', fontsize=14, c='goldenrod',
                 ha='center')
        plt.text(21.25, height - drag, 'Beta', fontsize=14, c=get_range(25),
                 ha='center')
        plt.text(40, height - drag, 'Gamma', fontsize=14, c=get_range(45),
                 ha='center')
    elif effect_size:
        plt.text(3.25, .05, 'Delta', fontsize=14, c=get_range(2.5),
                 ha='center')
        plt.text(5.75, .2, 'Theta', fontsize=14, c='green',
                 ha='center')
        plt.text(9.75, .05, 'Alpha', fontsize=14, c='goldenrod',
                 ha='center')
        plt.text(21.25, .05, 'Beta', fontsize=14, c=get_range(25),
                 ha='center')
        plt.text(40, .05, 'Gamma', fontsize=14, c=get_range(45),
                 ha='center')
    else:
        plt.text(3.25, .002, 'Delta', fontsize=14, c=get_range(2.5),
                 ha='center')
        plt.text(5.75, .008, 'Theta', fontsize=14, c='green',
                 ha='center')
        plt.text(9.75, .002, 'Alpha', fontsize=14, c='goldenrod',
                 ha='center')
        plt.text(21.25, .002, 'Beta', fontsize=14, c=get_range(25),
                 ha='center')
        plt.text(40, .002, 'Gamma', fontsize=14, c=get_range(45),
                 ha='center')

    plt.gca().spines[['right', 'top', ]].set_visible(False)
    plt.gcf().subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    frontal_str = '_frontal' if just_frontal else ''
    if effect_size:
        fp_out = fr'result_pics/Fig7/hz_effect_size{frontal_str}.png'
    else:
        last = 'VD' if flip_VD else 'PA'
        fp_out = fr'result_pics/Fig7/hz_corr_{last}.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()

    if flip_VD and not effect_size:
        plot_hz_corrs(name2r, effect_size=effect_size, t_vals=t_vals,
                      plot_se=plot_se, flip_VD=False)
