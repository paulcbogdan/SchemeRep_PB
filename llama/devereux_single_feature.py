import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mtick

from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_feature_regression import get_vecs_for_regr
from llama.devereux_llama import get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
from llama.plot_2rel import general_llama_plot
from marinate.pkld import pkld


def create_modified_turbo(remove_light_peak=True, remove_dark_ends=False):
    """
    Create a modified version of the turbo colormap.

    Parameters:
    remove_light_peak (bool): If True, removes the high lightness peak
    remove_dark_ends (bool): If True, removes the dark ends of the colormap
    """
    # Get the turbo colormap data
    turbo = plt.cm.turbo
    turbo_vals = turbo(np.linspace(0, 1, 256))

    if remove_light_peak:
        # Convert to HSL to modify lightness
        hsv = mcolors.rgb_to_hsv(turbo_vals[:, :3])
        # Reduce lightness peaks by scaling down high values
        mask = hsv[:, 2] > 0.75  # Adjust this threshold as needed
        hsv[mask, 2] *= 0.75  # Reduce the intensity of bright spots
        # Convert back to RGB
        modified_vals = mcolors.hsv_to_rgb(hsv)
    else:
        modified_vals = turbo_vals[:, :3]

    if remove_dark_ends:
        # Increase brightness at the ends
        ramp_length = 25  # Number of values to modify at each end

        # Modify start
        start_ramp = np.linspace(0.3, 1, ramp_length)
        modified_vals[:ramp_length] *= start_ramp[:, np.newaxis]

        # Modify end
        end_ramp = np.linspace(1, 0.3, ramp_length)
        modified_vals[-ramp_length:] *= end_ramp[:, np.newaxis]

    # Create new colormap
    return mcolors.ListedColormap(modified_vals)


@pkld(overwrite=False)
def regression_one_feature(feature='is_small',
                           pf_thresh=300,
                           cat='gate_proj_in', layer_name=8,
                           activation_model='meta-llama/Llama-3.2-3b',
                           # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                           normalize=False, quick=1, item_standard='deve',
                           position=0, symmetric=False,
                           req=5, normalize_regr=False,
                           do_r2=True
                           ):
    vecs = get_vecs_for_regr(activation_model, pf_thresh, item_standard, cat,
                             layer_name, normalize, quick, position, symmetric,
                             store_='both'
                             )

    items, _ = get_standard_items_list(pf_thresh, item_standard)
    items = items[:pf_thresh]
    feat2onehot = get_binary_feat_matrix(items, item_standard, threshold=10,
                                         pf_thresh=300, req=req)
    d_add = {}
    for feat, onehot in feat2onehot.items():
        d_add[feat[:15]] = onehot
    feat2onehot = {**feat2onehot, **d_add}

    try:
        onehot = feat2onehot[feature]
    except KeyError:
        print(f'{feature=} not found in {feat2onehot.keys()}')
        raise KeyError
    baseline = np.mean(onehot)
    baseline = np.max([baseline, 1 - baseline])

    res = fit_regularized_models(vecs, onehot, normalize=normalize_regr,
                                 do_r2=do_r2)
    return res['Ridge']['r2_score'], baseline


def get_accuracy_curves(cat, pf_thresh, threshold, req,
                        activation_model, do_r2='5050'):
    items, _ = get_standard_items_list(pf_thresh, 'deve')
    feat2onehot = get_binary_feat_matrix(items, 'deve', threshold=20,
                                         pf_thresh=pf_thresh, req=req)
    feats_og = list(feat2onehot)
    # feats_og += ['is_a_musical_in']
    feats_og += ['is_found_in_sea']
    feats = ['is_a_mammal', 'is_a_bird', 'is_worn', 'is_a_musical_in', 'has_feathers', 'is_an_insect', 'has_wings',
             'is_clothing', 'has_wheels', 'is_a_vegetable', 'is_eaten_edible', 'made_of_metal',
             'is_found_in_sea', 'made_of_fabric_', 'is_a_weapon', 'is_a_plant', 'has_fur_hair',
             'does_fly', 'does_swim', 'is_electric', 'is_found_in_kit', 'is_an_animal',
             'does_make_sound', 'has_a_handle_ha', 'does_carry_tran', 'made_of_wood', 'has_legs', 'is_pretty_attra',
             'is_a_tool', 'is_warm', 'has_a_tail', 'has_teeth', 'is_for_children',
             'made_of_plastic', 'has_claws', 'is_expensive',
             'is_circular_rou', 'does_smell_is_s', 'is_colourful',
             'is_thin', 'is_small', 'is_fast', 'made_of_glass', 'is_heavy', 'is_soft', 'is_strong']
    feats = [feat for feat in feats if feat in feats_og]
    feats = feats[:20]
    # 'is_food', too similar to is_edible
    # 'is_big_large', too similar to is_small
    # 'is_a_fruit', eh
    # 'is_dangerous',
    # 'has_skin_peel',
    accuracy_curves_l = []
    baselines = []
    for i, feat in enumerate(feats):  # [:5]:
        # print(f'Doing ({cat}): {feat=}')
        accuracy_curve = []
        for layer_name in range(80 if '70b' in activation_model else 28):
            acc, baseline = regression_one_feature(feat, cat=cat,
                                          layer_name=layer_name,
                                          activation_model=activation_model,
                                          do_r2=do_r2,
                                          # do_r2=False,
                                          pf_thresh=pf_thresh, req=req)
            accuracy_curve.append(acc)
            if layer_name == 0:
                baselines.append(baseline)
        accuracy_curves_l.append(accuracy_curve)
    return np.array(accuracy_curves_l), np.array(baselines), feats


def get_feat2name():
    d = {'is_an_animal': 'is animal', 'made_of_metal': 'made of metal',
         'has_wings': 'has wings', 'has_legs': 'has legs',
         'is_a_bird': 'is a bird', 'has_fur_hair': 'has fur/hair',
         'has_feathers': 'has feathers', 'made_of_wood': 'made of wood',
         'does_fly': 'does fly', 'is_a_mammal': 'is a mammal',
         'is_a_tool': 'is a tool',  # 'is_food':
         # 'is food',
         'is_eaten_edible': 'is edible',
         'is_dangerous': 'is dangerous', 'is_a_fruit': 'is a fruit',
         'made_of_plastic': 'made of plastic', 'is_big_large': 'is_large',
         'has_a_tail': 'has a tail', 'is_expensive': 'is expensive',
         'is_heavy': 'is heavy', 'is_soft': 'is_soft', 'is_fast': 'is_fast',
         'is_small': 'is small', 'is_a_musical_in': 'is a musical instrument',
         'is_found_in_sea': 'found in sea'}
    return d


def plot_all_feats(  # cat='input',
        # cat='down_proj_out',
        # cat='attn_output',
        cat='gate_proj_in',
        # activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
        # activation_model='meta-llama/Llama-3.3-70b-Instruct',
        pf_thresh=300, threshold=20, req=5,
        activation_model='meta-llama/Llama-3.2-3b',
        feat2color=None,
        xlabel=False,
        do_r2=False
):
    accuracy_curves_l, baselines, feats = (
        get_accuracy_curves(cat, pf_thresh, threshold, req, activation_model,
                            do_r2=do_r2))
    if do_r2 == '5050':
        baselines = np.full_like(baselines, 0.5)

    norm = plt.Normalize(vmin=0, vmax=len(feats))
    cmap_t = plt.get_cmap('turbo')
    cmap = create_modified_turbo(remove_light_peak=True, remove_dark_ends=False)

    accuracy_curves_l -= baselines[:, None]
    last_accs = accuracy_curves_l[:, -1]
    acc_order = np.argsort(last_accs)
    accuracy_curves_l = accuracy_curves_l[acc_order]
    feats = [feats[i] for i in acc_order]
    feat2name = get_feat2name()
    feats = [feat2name[feat] for feat in feats]
    last_accs = [last_accs[i] for i in acc_order]

    if feat2color is None:
        c_darks = [cmap(norm(i)) for i in range(len(feats))]
        c_darks = c_darks[0::4] + c_darks[1::4] + c_darks[2::4] + c_darks[3::4]
        c_lights = [cmap_t(norm(i)) for i in range(len(feats))]
        c_lights = c_lights[0::4] + c_lights[1::4] + c_lights[2::4] + c_lights[3::4]
        feat2color = {feat: (c_dark, c_light) for feat, c_dark, c_light in
                      zip(feats, c_darks, c_lights)}
    else:
        c_darks = [feat2color[feat][0] for feat in feats]
        c_lights = [feat2color[feat][1] for feat in feats]

    for accuracy_curve, feat, c_dark, c_light in (
            zip(accuracy_curves_l, feats, c_darks, c_lights)):
        plt.plot(accuracy_curve, label=feat, alpha=0.5,
                 marker='o', markersize=3., color=c_dark,
                 linewidth=1)
        plt.plot(accuracy_curve, label=feat, alpha=1,
                 marker='o', markersize=1., color=c_light,
                 linewidth=0.25)

    acc_min = min(last_accs)
    acc_max = max(last_accs)
    if do_r2 == '5050':
        ylim_low = -0.075
        ylim_high = 0.5
    else:
        ylim_low = -0.075
        ylim_high = 0.29
    gap = ylim_high - ylim_low
    acc_min -= gap * .025
    acc_max += gap * .025

    if '3b' in activation_model or '3b' in activation_model[0]:
        num_layers = 28
    else:
        num_layers = 80
    # print(f'{num_layers=}')
    # quit()

    for c, feat, last_acc, spot, in zip(c_darks, feats, last_accs,
                                        np.linspace(acc_min, acc_max, len(c_darks))):
        plt.text(num_layers + 1, spot, feat, fontsize=10, color=c, va='center')
        plt.plot([num_layers - 1, num_layers + 0.8],
                 [last_acc, spot], color=c, linewidth=0.5,
                 alpha=.5, zorder=-1)

    plt.ylabel('Accuracy relative to baseline (Δ%)')
    if xlabel: plt.xlabel('Layer')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    cat2title = {'input': 'Residual stream (transformer input)',
                 'attn_output': 'Attention layer output',
                 'down_proj_out': 'FFN output', }
    plt.title(cat2title[cat])
    # plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.xlim(-0.5, num_layers + 0.5)
    plt.ylim(ylim_low, ylim_high)
    if num_layers == 28:
        plt.xticks(np.arange(0, num_layers, 5))
    else:
        plt.xticks(np.arange(0, num_layers, 10))
    if do_r2 == '5050':
        plt.yticks([0.0, 0.1, 0.2, .3, .4, .5])
    else:
        plt.yticks([-0.05, 0, 0.05, 0.1, .15, .2, .25])
    plt.gca().spines[['top', 'right']].set_visible(False)
    return feat2color


def plot_each_layer_component():
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)

    pass


def plot_triangle(  # activation_model='meta-llama/Llama-3.2-3b',
        activation_model='meta-llama/Llama-3.3-70b-Instruct',
):
    # Create figure
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(13, 10))

    # Create a 2x4 grid
    gs = fig.add_gridspec(2, 6)

    # Top plot spanning two columns (positions 1 and 2)
    ax1 = fig.add_subplot(gs[0, 1:5])
    plt.sca(ax1)
    feat2color = plot_all_feats(cat='input', activation_model=activation_model)

    # Bottom plots - each spanning two columns
    ax2 = fig.add_subplot(gs[1, 0:3])
    plt.sca(ax2)
    plot_all_feats(cat='attn_output', activation_model=activation_model,
                   feat2color=feat2color)

    if activation_model != r'meta-llama/Llama-3.3-70b-Instruct':
        ax3 = fig.add_subplot(gs[1, 3:])
        plt.sca(ax3)
        plot_all_feats(cat='down_proj_out', activation_model=activation_model,
                       feat2color=feat2color)

    # Adjust layout to prevent overlap
    # plt.tight_layout()
    plt.subplots_adjust(wspace=50, left=0.1, right=0.9, top=0.95, bottom=0.10)

    # Show plot
    plt.show()
    quit()


def plot_square(activation_model='meta-llama/Llama-3.2-3b', do_r2='5050',
                bury=False, do_legend=True, just2=False, suptitle=None):
    plt.rcParams.update({'font.size': 14})
    if bury:
        activation_model = (activation_model, 'bury_item')

    if just2:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        plt.sca(axs[0])
    else:
        fig, axs = plt.subplots(2, 2, figsize=(13, 10))
        plt.sca(axs[0, 0])
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=16)
    single_feat_multi_cats(pf_thresh=300, threshold=20, req=5,
                           activation_model=activation_model,
                           do_r2=do_r2, do_legend=do_legend)
    plt.gca().text(-0.2, 1.04, 'a.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    if just2:
        plt.sca(axs[1])
    else:
        plt.sca(axs[0, 1])
    feat2color = plot_all_feats(cat='input', activation_model=activation_model,
                                do_r2=do_r2)
    plt.gca().text(-0.2, 1.04, 'b.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')
    if just2:
        if suptitle is not None:
            plt.subplots_adjust(wspace=0.5, left=0.1, right=0.9, top=0.85, bottom=0.11,
                                hspace=0.2)
        else:
            plt.subplots_adjust(wspace=0.5, left=0.1, right=0.9, top=0.93, bottom=0.11,
                                hspace=0.2)
        plt.show()
        return

    plt.sca(axs[1, 0])
    plot_all_feats(cat='attn_output', activation_model=activation_model,
                   feat2color=feat2color, xlabel=True, do_r2=do_r2)
    plt.gca().text(-0.2, 1.04, 'c.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    # if '3b' in activation_model or '3b' in activation_model[0]:
    plt.sca(axs[1, 1])
    plot_all_feats(cat='down_proj_out', activation_model=activation_model,
                   feat2color=feat2color, xlabel=True, do_r2=do_r2)
    plt.gca().text(-0.2, 1.04, 'd.', transform=plt.gca().transAxes,
             fontsize=18, fontweight='bold', va='center')

    plt.subplots_adjust(wspace=0.5, left=0.1, right=0.9, top=0.95, bottom=0.07,
                        hspace=0.2)
    plt.show()


def single_feat_multi_cats(pf_thresh=300, threshold=20, req=5,
                           activation_model='meta-llama/Llama-3.2-3b',
                           do_r2='5050', do_legend=True):
    cat2vals = {}
    cats = ['input', 'attn_output', 'down_proj_out', #'attn_weights'
            ]
    if '70b' in activation_model or '70b' in activation_model[0]:
        # cats = cats[:2] + cats[3:]
        num_layers = 80
    else:
        num_layers = 28
    # print(cats)
    # quit()
    #     cats = [cat for cat in cats if cat != 'down_proj_out']

    for cat in cats:
        accuracy_carves, baseline, _ = (
            get_accuracy_curves(cat, pf_thresh, threshold, req, activation_model,
                                do_r2=do_r2))
        if do_r2 == '5050':
            baseline = np.full_like(baseline, 0.5)
        accuracy_carves -= baseline[:, None]
        cat2vals[cat] = np.nanmean(accuracy_carves, axis=0)
    cat2vals['drop_proj_out'] = np.full(num_layers, np.nan)
    cats.append('drop_proj_out')

    vals_l = [cat2vals[cat] for cat in cats]
    labels = ['Residual\n(input)', 'Attention\naddition',
              'FFN\naddition', #'Attention\nweights'
              ]
    colors = ['k', 'r', 'dodgerblue', #'green'
              ]
    cats = cats[:1]
    labels = labels[:1]
    colors = colors[:1]
    # if '70b' in activation_model or '70b' in activation_model[0]:
    #     labels = labels[:2] + labels[3:]
    #     colors = colors[:2] + colors[3:]

    plt.title('Averages across 20 item features')
    general_llama_plot(vals_l, labels, colors,
                       ylabel='Accuracy relative to baseline (Δ%)',
                       y_low=-0.0025, y_high=0.5 if do_r2 == '5050' else 0.08,
                       xlabel=False, num_layers=num_layers, do_legend=do_legend)


if __name__ == '__main__':
    # for layer_name in range(53, 80):
    #     test = regression_one_feature('is_fast', cat='attn_output',
    #                                   layer_name=layer_name,
    #                                   activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                                   # do_r2='5050',
    #                                   do_r2=False,
    #                                   pf_thresh=300, req=5)
    #     print(test)
    # quit()
    # plot_square(activation_model=('meta-llama/Llama-3.2-3b', 'bury_item'))

    plot_square()
    # plot_square(activation_model='meta-llama/Llama-3.3-70b-Instruct')
    # single_feat_multi_cats()
    # plot_all_feats(cat='down_proj_out', activation_model='meta-llama/Llama-3.2-3b')
    # plot_all_feats(cat='input', activation_model='meta-llama/Llama-3.2-3b')
    # plot_all_feats(cat='attn_output', activation_model='meta-llama/Llama-3.2-3b')
    # plt.show()
    # quit()

    # plot_all_feats(cat='down_proj_out', activation_model=('meta-llama/Llama-3.3-70b-Instruct',
    #                                                       'bury_solo'))
    # TODO: run this 100% for down_proj_out
