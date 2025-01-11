import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def general_llama_plot(vals_l, labels, colors, ylabel='Accuracy (%)',
                       y_low=0.5, y_high=1, num_layers=28, xlabel=True,
                       do_legend=True):
    # plt.rcParams.update({'font.size': 14})
    # fig = plt.figure(figsize=(5, 4))
    for vals, label, c in zip(vals_l, labels, colors):
        vals = [0.5 if np.isnan(val) else val for val in vals]
        plt.plot(list(range(len(vals))), vals,
                 label=label, color=c, marker='.',
                 linewidth=1)
    plt.ylim(y_low, y_high)
    if '%' in ylabel:
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.xlim(-0.5, num_layers + 0.5)
    plt.ylabel(ylabel, labelpad=10 if '\n' in ylabel else 5)
    if xlabel: plt.xlabel('Layer')
    plt.gca().spines[['top', 'right']].set_visible(False)
    if do_legend:
        legend = plt.legend(loc='center left', fontsize=12, ncol=1,
                   bbox_to_anchor=(0.95, 0.5),
                   frameon=False, columnspacing=0.35, handletextpad=0.5,
                   markerscale=1, handlelength=1.5,
                   scatteryoffsets=[0.55], scatterpoints=1)
    # for text in legend.get_texts():
    #     text.set_ha('center')
    plt.xticks(np.arange(0, num_layers, 5))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.15)
    # plt.show()
