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


def cross_accs(cat='attn_output', st_layer=0, gap=0,
               # activation_model='meta-llama/Llama-3.3-70b-Instruct'
               activation_model='meta-llama/Llama-3.2-3b',
               ):
    # cat = 'down_proj_out'
    experiments = ['SchemeRep', 'SchemeRep_deve', 'SchemeRep_buried',
                   # 'SchemeRep_bured_deve',
                   'Rissman', 'Rissman_buried',
                   'carn_herb', 'analogy_within', 'analogy_within_buried']
    # experiments = ['analogy_within', 'analogy_within_buried']
    difs = []
    for experiment in experiments:
        vals = get_accs(experiment=experiment, cat=cat,
                        activation_model=activation_model)
        exp_str = f'{experiment:<22}: '
        for i in range(1, 5):
            r = get_autocorr(vals, gap=i)
            exp_str += f'r[{i}] = {r:+.2f} | '
        exp_str = exp_str[:-3]
        print(exp_str)
        # r = get_autocorr(vals)
        # r2 = get_autocorr(vals, gap=2)
        # print(f'{experiment}: {r=:.2f} | {r2=:.2f}')

        vals = vals[st_layer:]
        dif0 = np.diff(vals, n=gap)
        difs.append(dif0)
    difs = np.array(difs).T
    difs = stats.zscore(difs, axis=0, nan_policy='omit')

    layers = np.arange(st_layer, st_layer + difs.shape[0])
    for dif, experiment in zip(difs.T, experiments):
        plt.plot(layers, dif, label=experiment, marker='o', markersize=2,
                 alpha=.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
               frameon=False)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    # quit()

    # df = pd.DataFrame(difs, columns=experiments)
    # print(df.corr())
    corr = np.corrcoef(difs, rowvar=False)
    corr[np.diag_indices_from(corr)] = np.nan
    plot_heatmap(corr, experiments, vmin=-1, vmax=1)
    # plt.imshow(corr, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    # plt.show()


def get_autocorr(l, gap=1, st_pt=14):
    l = l[st_pt:]
    dif0 = np.diff(l, n=gap)
    return stats.spearmanr(dif0[:-gap], dif0[gap:], nan_policy='omit')[0]


if __name__ == '__main__':
    cross_accs()
