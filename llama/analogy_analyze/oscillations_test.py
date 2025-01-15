from functools import partial

import numpy as np
import pandas as pd
from scipy import stats

from llama.Rissman_similarity_analysis import analyze_rissman
from llama.analogy_analyze.analogy_llama import run_analogy_analysis
from llama.carnivore_herbivore import cross_species_regression
from llama.model_comparison import plot_heatmap
from marinate.pkld import pkld
import matplotlib.pyplot as plt


@pkld(overwrite=False)
def get_accs(activation_model='meta-llama/Llama-3.3-70b-Instruct',
             cat='attn_output', experiment='SchemeRep'):
    kw = {'activation_model': activation_model, 'cat': cat}
    if experiment == 'SchemeRep':
        kw_add = {'do_SchemeRep': True, 'norm_SchemeRep': True,
                  'binary_nonrep': False, 'no_neu': (False, 'cont')}
        kw.update(kw_add)
        f = partial(analyze_rissman, **kw)
    elif experiment == 'SchemeRep_deve':
        kw_add = {'do_SchemeRep': (True, 'deve'), 'norm_SchemeRep': True,
                  'binary_nonrep': False, 'no_neu': (False, 'cont')}
        kw.update(kw_add)
        f = partial(analyze_rissman, **kw)
    elif experiment == 'SchemeRep_buried':
        kw['activation_model'] = (activation_model, 'bury')
        kw_add = {'do_SchemeRep': True, 'norm_SchemeRep': True,
                  'binary_nonrep': False, 'no_neu': (False, 'cont')}
        kw.update(kw_add)
        f = partial(analyze_rissman, **kw)
    # elif experiment == 'SchemeRep_bured_deve':
    #     kw['activation_model'] = (activation_model, 'bury')
    #     kw_add = {'do_SchemeRep': (True, 'deve'), 'norm_SchemeRep': True,
    #               'binary_nonrep': False, 'no_neu': (False, 'cont')}
    #     kw.update(kw_add)
    #     f = partial(analyze_rissman, **kw)
    elif experiment == 'Rissman':
        kw_add = {'do_SchemeRep': False, 'norm_SchemeRep': False,
                  'binary_nonrep': False, 'no_neu': False}
        kw.update(kw_add)
        f = partial(analyze_rissman, **kw)
    elif experiment == 'Rissman_buried':
        kw['activation_model'] = (activation_model, 'bury')
        kw_add = {'do_SchemeRep': False, 'norm_SchemeRep': False,
                  'binary_nonrep': False, 'no_neu': False}
        kw.update(kw_add)
        f = partial(analyze_rissman, **kw)
    elif experiment == 'carn_herb':
        kw_add = {'cross_animal': True, 'food_second': 'both'}
        kw.update(kw_add)
        f = partial(cross_species_regression, **kw)
        pass
    elif experiment == 'analogy_within':
        kw_add = {'activation_model': activation_model, 'position': 3,
                  'do_r2': False, 'copies': (0, 1), 'flip_within': True,
                  'analogy': 1}
        kw.update(kw_add)
        f = partial(run_analogy_analysis, **kw)
    elif experiment == 'analogy_within_buried':
        kw['activation_model'] = (activation_model, 'bury')
        kw_add = {'position': 3, 'do_r2': False, 'copies': (0, 1),
                  'flip_within': True, 'analogy': 1}
        kw.update(kw_add)
        f = partial(run_analogy_analysis, **kw)
    else:
        raise ValueError

    num_layers = 28 if ('3b' in activation_model or '3b' in activation_model[0]) else 80
    vals = []
    for layer in range(num_layers):
        vals.append(f(layer_name=layer))
    return vals


def cross_accs(#cat='attn_output',
               cat='down_proj_out',
               st_layer=0, gap=0,
               activation_model='meta-llama/Llama-3.3-70b-Instruct'
               #activation_model='meta-llama/Llama-3.2-3b',
               ):
    # cat = 'down_proj_out'
    experiments = ['SchemeRep', 'Rissman', 'carn_herb',
                   'SchemeRep_buried', 'Rissman_buried', 'analogy_within_buried',
                   'analogy_within',]
    labels = ['Exp. 2A', 'Exp. 2B', 'Exp. 2C',
              'Exp. 2A (buried)', 'Exp. 2B (buried)', 'Exp. 3 (hard, buried)',
              'Exp. 3 (hard)',]

    colors = ['darkred', 'red', 'lightcoral',
              'darkblue', 'dodgerblue', 'lightseagreen',
              'k',]

    # experiments = ['analogy_within', 'analogy_within_buried']
    difs = []
    d_r = {}
    for i in range(5):
        d_r[i] = []

    for experiment in experiments:
        vals = get_accs(experiment=experiment, cat=cat,
                        activation_model=activation_model)
        exp_str = f'{experiment:<22}: '
        for i in range(5):
            r = get_autocorr(vals, gap=i)
            d_r[i].append(r)
            exp_str += f'r[{i}] = {r:+.2f} | '
        exp_str = exp_str[:-3]
        print(exp_str)

        vals = vals[st_layer:]
        dif0 = np.diff(vals, n=gap)
        difs.append(dif0)
    exp_str = 'Mean autocorrs'
    print(f'{cat=}')
    for i in range(5):
        M_r = np.mean(d_r[i])
        SD_r = np.std(d_r[i])
        exp_str += f'r[{i}] = {M_r:+.2f} ({SD_r:.2f}) | '
    exp_str = exp_str[:-3]
    print(exp_str)
    # quit()

    difs = np.array(difs).T
    difs = stats.zscore(difs, axis=0, nan_policy='omit')

    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(6, 4))

    if gap == 0:
        title = 'Accuracies'
        if cat == 'attn_output':
            title += ' (attention layer outputs)'
        elif cat == 'down_proj_out':
            title += ' (FFN layer outputs)'
        letter = 'a.'
    elif gap == 1:
        title = 'Discrete first derivatives'
        if cat == 'attn_output':
            title += ' (attention layer outputs)'
        elif cat == 'down_proj_out':
            title += ' (FFN layer outputs)'
        letter = 'b.'
    plt.title(title)

    layers = np.arange(st_layer, st_layer + difs.shape[0])
    for dif, experiment, label, color in (
            zip(difs.T, experiments, labels, colors)):
        plt.plot(layers, dif, label=label, marker='o', markersize=2,
                 alpha=.7, color=color, linewidth=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 6, 1, 4, 2, 5]
    handles_ = [handles[i] for i in order]
    labels_ = [labels[i] for i in order]

    plt.legend(handles_, labels_,
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
               frameon=False,)
    plt.ylabel('Z-scored accuracy difference', fontsize=11)
    plt.xlabel('Layer', fontsize=11)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(-1, 80)
    # plt.tight_layout()  # Adjust layout to prevent overlap

    plt.gca().text(-0.10, 1.05, letter, transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', va='center')

    plt.subplots_adjust(left=0.1, right=0.93, top=0.93, bottom=0.3)
    plt.show()
    # quit()
    #
    # # df = pd.DataFrame(difs, columns=experiments)
    # # print(df.corr())

    # labels_ = [label.replace(' (hard, buried', '\n(hard,\nburied') for label in labels]
    labels_ = [label.replace(' (buried', '\n(buried') for label in labels]
    labels_ = [label.replace(' (hard', '\n(hard') for label in labels_]

    plt.rcParams.update({'font.size': 12})
    labels_hor = ['2A', '2B', '2C',
                  '2A\n(b.)', '2B\n(b.)', '3\n(b.)',
                  '3',]
    if gap >= 1:
        title = 'First derivatives\ncorrelation matrix'# (last 40 layers)'
        # difs = difs[-40:]
    else:
        title = 'Correlation matrix'

    corr = np.corrcoef(difs, rowvar=False)
    corr[np.diag_indices_from(corr)] = np.nan
    M_corr = np.nanmean(corr)
    SD_corr = np.nanstd(corr)
    print(f'{M_corr=:.2f} [{SD_corr=:.2f}]')

    plot_heatmap(corr, labels_, labels_hor,
                 vmin=-0.6, vmax=0.6, rotation=0,
                 title=title)#, cmap='seismic')


def get_autocorr(l, gap=1, st_pt=0):
    if gap == 0:
        l = l[st_pt:]
        return stats.spearmanr(l[:-1], l[1:], nan_policy='omit')[0]
    else:
        l = l[st_pt:]
        dif0 = np.diff(l, n=gap)
        return stats.spearmanr(dif0[:-gap], dif0[gap:],
                               nan_policy='omit')[0]


if __name__ == '__main__':
    # cross_accs(cat='down_proj_out', gap=0)
    # cross_accs(cat='down_proj_out', gap=1)
    print('-')
    cross_accs(cat='attn_output', gap=0)
    cross_accs(cat='attn_output', gap=1)
