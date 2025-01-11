import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_feature_regression import get_vecs_for_regr
from llama.devereux_llama import get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
from marinate.pkld import pkld
import matplotlib.colors as mcolors

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

@pkld
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
    res = fit_regularized_models(vecs, onehot, normalize=normalize_regr,
                                 do_r2=do_r2)
    return res['Ridge']['r2_score']

def get_accuracy_curves(cat, pf_thresh, threshold, req,
                        activation_model):
    items, _ = get_standard_items_list(pf_thresh, 'deve')
    feat2onehot = get_binary_feat_matrix(items, 'deve', threshold=20,
                                         pf_thresh=pf_thresh, req=req)
    feats_og = list(feat2onehot)
    feats = ['is_a_mammal', 'is_a_bird', 'is_worn', 'is_a_musical_in', 'has_feathers', 'is_an_insect', 'has_wings',
             'is_clothing', 'has_wheels', 'is_a_vegetable', 'is_eaten_edible', 'is_a_fruit', 'made_of_metal',
             'is_found_in_sea', 'made_of_fabric_', 'has_skin_peel', 'is_a_weapon', 'is_a_plant', 'has_fur_hair',
             'does_fly', 'does_swim', 'is_electric', 'is_found_in_kit', 'is_an_animal', 'is_dangerous',
             'does_make_sound', 'has_a_handle_ha', 'does_carry_tran', 'made_of_wood', 'has_legs', 'is_pretty_attra',
             'is_a_tool', 'is_warm', 'has_a_tail', 'has_teeth', 'is_food', 'is_big_large', 'is_for_children',
             'made_of_plastic', 'has_claws', 'is_expensive', 'is_circular_rou', 'does_smell_is_s', 'is_colourful',
             'is_thin', 'is_small', 'is_fast', 'made_of_glass', 'is_heavy', 'is_soft', 'is_strong']
    feats = feats[:50]
    feats = [feat for feat in feats if feat in feats_og]
    feats = feats[:20]
    print(f'{len(feats)=}')


    accuracy_curves_l = []
    for i, feat in enumerate(feats):  # [:5]:
        accuracy_curve = []
        for layer_name in range(80 if '70b' in activation_model else 28):
            acc = regression_one_feature(feat, cat=cat,
                                        layer_name=layer_name,
                                        activation_model=activation_model,
                                        # do_r2='5050',
                                        do_r2=False,
                                        pf_thresh=pf_thresh, req=req)
            accuracy_curve.append(acc)
        accuracy_curves_l.append(accuracy_curve)
    return np.array(accuracy_curves_l), feats
def plot_all_feats(#cat='input',
                   #cat='down_proj_out',
                   # cat='attn_output',
                   cat='gate_proj_in',
                   # activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
                   # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                   pf_thresh=300, threshold=20, req=5,
                   activation_model='meta-llama/Llama-3.2-3b',
                   feat2color=None
                   ):
    accuracy_curves_l, feats = (
        get_accuracy_curves(cat, pf_thresh, threshold, req, activation_model))

    norm = plt.Normalize(vmin=0, vmax=len(feats))
    cmap_t = plt.get_cmap('turbo')
    cmap = create_modified_turbo(remove_light_peak=True, remove_dark_ends=False)

    last_pts = []
    accuracy_curves_l -= accuracy_curves_l[:, 0][:, None]
    last_accs = accuracy_curves_l[:, -1]
    acc_order = np.argsort(last_accs)
    accuracy_curves_l = accuracy_curves_l[acc_order]
    feats = [feats[i] for i in acc_order]

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

    for accuracy_curve, feat, c_dark, c_light in zip(accuracy_curves_l, feats,
                                                     c_darks, c_lights):
        plt.plot(accuracy_curve, label=feat, alpha=0.5,
                 marker='o', markersize=3., color=c_dark,
                 linewidth=1)
        plt.plot(accuracy_curve, label=feat, alpha=1,
                 marker='o', markersize=2., color=c_light,
                 linewidth=0.25)

    acc_min = min(last_accs)
    acc_max = max(last_accs)
    for c, feat, spot in zip(c_darks, feats,
                       np.linspace(acc_min, acc_max, len(c_darks))):
        plt.text(28, spot, feat, fontsize=8, color=c)

    from matplotlib import ticker as mtick
    # plt.yticks(fontsize=12)
    plt.ylabel('Accuracy to relative baseline (Δ%)')
    plt.xlabel('Layer')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    # plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.xlim(-0.5, 28.5)
    plt.gca().spines[['top', 'right']].set_visible(False)
    return feat2color

    # quit()
    # r2s_all = np.array(r2s_all)
    # # r2s_all -= np.nanmean(r2s_all, axis=1)[:, None]
    # # r2s_all -= r2s_all[:, 0][:, None]
    # r2s_M = np.nanmean(r2s_all, axis=0)
    # print(f'{r2s_all.shape=}')
    #
    # confidence_interval = np.std(r2s_all, axis=0) / np.sqrt(r2s_all.shape[0])
    #
    # plt.plot(r2s_M, alpha=0.5,
    #          marker='o', markersize=2,
    #          color='blue')
    # plt.fill_between(list(range(r2s_M.shape[0])),
    #                  r2s_M - confidence_interval,
    #                  r2s_M + confidence_interval,
    #                  color='dodgerblue', alpha=0.2,
    #                  # label='Confidence Interval'
    #                  )
    # plt.title(f'{cat=}, {activation_model=}')
    # # plt.legend()
    # plt.show()

def plot_each_layer_component():
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)

    pass

def plot_triangle():
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    # Create figure
    plt.rcParams.update({'font.size': 20})

    fig = plt.figure(figsize=(13, 10))

    # Create a 2x4 grid
    gs = fig.add_gridspec(2, 4)

    # Top plot spanning two columns (positions 1 and 2)
    ax1 = fig.add_subplot(gs[0, 1:3])
    plt.sca(ax1)
    plot_all_feats(cat='input')

    # Bottom plots - each spanning two columns
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.plot(x, y2)
    ax2.set_title('Bottom Left')

    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.plot(x, y3)
    ax3.set_title('Bottom Right')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == '__main__':
    plot_triangle()
    # plot_all_feats()
    # TODO: run this 100% for down_proj_out
