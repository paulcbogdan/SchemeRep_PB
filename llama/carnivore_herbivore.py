import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.analogy_analyze.analogy_llama import prepare_analogy_figs
from llama.devereux_llama import get_llama_activations_deve
from llama.get_obj_scn_vecs import process_cat_cat_inner
from llama.plot_2rel import general_llama_plot
from marinate.pkld import pkld
from matplotlib import ticker as mtick


# suppress: RuntimeWarning: invalid value encountered in divide
np.seterr(divide='ignore', invalid='ignore')

def make_df_carnivore_herbivore(activation_model='meta-llama/Llama-3.2-3b',
                                cat='gate_proj_in', layer_name=1,
                                do_unrelated=False, food_second='both'):
    carnivores = ['shark', 'lion', 'wolf', 'tiger', 'bear',
                  'alligator', 'eagle', 'vulture', 'hyena', 'cougar']
    herbivores = [  # 'gazelle',
        'zebra', 'elephant', 'giraffe', 'hippo', 'panda',
        'turtle', 'rabbit', 'koala', 'whale', 'duck'
        # 'rhino', 'buffalo', 'koala','kangaroo'
    ]
    animals = carnivores + herbivores

    meats = ['salmon', 'chicken', 'mice', 'deer', 'steak',
             'tuna', 'lamb', 'egg', 'pigeon', 'pork']
    plants = ['leaves', 'wheat', 'corn', 'almond', 'bamboo',
              'grass', 'seeds', 'algae', 'apple', 'carrot']

    foods = meats + plants

    if do_unrelated:
        unrelated = ['ambulance', 'armchair', 'arrow', 'axe', 'bed'
                                                              'bicycle', 'blender', 'canary', 'car', 'cello',
                     'goggles', 'gun', 'washing_machine', 'yoyo', 'motorcycle',
                     'television', 'train', 'razor', 'rake', 'raft']
        foods = unrelated
        meats = unrelated[:10]
        plants = unrelated[10:]

    cat_, inner = process_cat_cat_inner(cat)
    d_vecs = {}
    for food in tqdm(foods, desc='Looping foods'):
        for animal in animals:
            for flip in [False, True]:
                if food_second == 'both':
                    pass
                elif food_second:
                    if not flip:
                        continue
                else:
                    if flip:
                        continue
                item0 = animal if flip else food
                item1 = food if flip else animal
                res, fp = pickle_wrap(get_llama_activations_deve,
                                      kwargs={'item0': item0, 'item1': item1,
                                              'activation_model': activation_model},
                                      easy_override=False, verbose=-1, dir_branches=1000,
                                      RAM_cache=True, get_fp=True)
                # print(F'{food}, {animal}')
                # print(f'{fp=}')
                # print('mlp_out' in res)
                # print('down_proj' in res['mlp_out'])
                # if cat == 'down_proj_out' and not (
                #         ('mlp_out' in res) and 'down_proj' in res['mlp_out']):
                #     res, fp = pickle_wrap(get_llama_activations_deve,
                #                           kwargs={'item0': item0, 'item1': item1,
                #                                   'activation_model': activation_model},
                #                           easy_override=True, verbose=-1, dir_branches=1000,
                #                           RAM_cache=True, get_fp=True)
                #     print(f'Attempt again: {cat=}')
                #     print(f'{fp=}')


                if 'mlp_out' in res:# and 'up_proj' in res['mlp_out']:
                    if 'up_proj' in res['mlp_out']:
                        del res['mlp_out']['up_proj']
                    if 'gate_proj' in res['mlp_out']:
                        del res['mlp_out']['gate_proj']
                    if 'down_proj' in res['mlp_in']:
                        del res['mlp_in']['down_proj']
                    if 'up_proj' in res['mlp_in']:
                        del res['mlp_in']['up_proj']
                    if 'act_fn' in res['mlp_in']:
                        del res['mlp_in']['act_fn']
                    if 'q_proj' in res['attn']:
                        del res['attn']['q_proj']
                    if 'k_proj' in res['attn']:
                        del res['attn']['k_proj']

                    with open(fp, 'wb') as f:
                        pickle.dump(res, f)

                # if isinstance(res, dict):
                #     for inner_, d in res.items():
                #         if inner_ in ['mlp_out', 'mlp_in', 'attn']:
                #             del_keys = []
                #             for outer in d.keys():
                #                 if outer != cat_:
                #                     del_keys.append(outer)
                #             for outer in del_keys:
                #                 del res[inner_][outer]

                for idx_target in [0, 1]:
                    if activation_model in ['BERT', 'simCSE']:
                        v = res[idx_target][layer_name]
                    elif cat in 'attn_weights':
                        v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                                       axis=(0, 1))
                    else:
                        assert len(res[inner][cat_][layer_name]) == 2
                        v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
                        if len(v.shape) > 1:
                            v = v.reshape(-1)
                    if idx_target == 0:  # sentence f'{item 0} and {item 1}'. focus on embedding: item0
                        if (item0, item1, item0) in d_vecs:
                            raise ValueError
                        d_vecs[(item0, item1, item0)] = v
                    else:
                        if (item0, item1, item1) in d_vecs:
                            raise ValueError
                        d_vecs[(item0, item1, item1)] = v
    return d_vecs, carnivores, herbivores, meats, plants


def get_animals_vecs(animals, foods, d_vecs, reverse=False,
                     get_food=False):
    vecs_all = []
    for animal in animals:
        vecs = []
        for food in foods:
            if get_food:
                if reverse:
                    vecs.append(d_vecs[(food, animal, food)])
                else:
                    vecs.append(d_vecs[(animal, food, food)])
            else:
                if reverse:
                    vecs.append(d_vecs[(animal, food, animal)])
                else:
                    vecs.append(d_vecs[(food, animal, animal)])
        M_vec = np.nanmean(np.array(vecs), axis=0)
        vecs_all.append(M_vec)
    return np.array(vecs_all)


def do_animal_food_animal_food(layer_name=4, reverse=False, get_food=False):
    d_vecs, carnivores, herbivores, meats, plants = (
        pickle_wrap(make_df_carnivore_herbivore,
                    kwargs={'layer_name': layer_name,
                            'do_unrelated': False,
                            'activation_model': 'meta-llama/Llama-3.2-3b',
                            # 'cat': 'attn_output',
                            },
                    easy_override=False, verbose=-1))

    animals = carnivores + herbivores
    foods = meats + plants

    vecs_all = []
    yes_eat = []
    for animal in animals:
        animal_l = []
        for food in foods:
            if reverse:
                animal_l.append(d_vecs[(food, animal, food)])
            else:
                animal_l.append(d_vecs[(animal, food, food)])
        animal_M = np.nanmean(np.array(animal_l), axis=0)

        for food in foods:
            if animal in carnivores and food in meats:
                yes_eat.append(1)
            elif animal in herbivores and food in plants:
                yes_eat.append(1)
            else:
                yes_eat.append(0)
            if reverse:
                vecs_all.append(d_vecs[(food, animal, food)])  # - animal_M)
            else:
                vecs_all.append(d_vecs[(animal, food, food)])  # - animal_M)
        # vecs_all.append(vecs)
    vecs_all = np.array(vecs_all)
    bad_cols = np.isnan(vecs_all).any(axis=0)
    vecs_all = vecs_all[:, ~bad_cols]
    vecs_all = stats.rankdata(vecs_all, axis=1)
    RSM_llama = np.corrcoef(vecs_all)

    yes_eat = np.array(yes_eat)
    RSM_eat = np.abs(yes_eat[:, None] - yes_eat[None, :])
    trils = np.tril_indices(RSM_llama.shape[0], -1)
    r, p = stats.spearmanr(RSM_llama[trils], RSM_eat[trils])
    print(f'{layer_name} | llama x eat: {r=:.3f} | {p=:.3f}')


def do_carnivore_herbivore(layer_name=4, reverse=False, get_food=False):
    d_vecs, carnivores, herbivores, meats, plants = (
        pickle_wrap(make_df_carnivore_herbivore,
                    kwargs={'layer_name': layer_name,
                            'do_unrelated': False,
                            # 'cat': 'attn_output'
                            },
                    easy_override=False, verbose=-1))

    vecs_carn = get_animals_vecs(carnivores, meats,  # + plants, #  meats
                                 d_vecs, reverse=reverse,
                                 get_food=get_food)
    vecs_herb = get_animals_vecs(herbivores, meats,  # plants
                                 d_vecs, reverse=reverse,
                                 get_food=get_food)

    vecs_all = np.concatenate([vecs_carn, vecs_herb], axis=0)
    bad_cols = np.isnan(vecs_all).any(axis=0)
    prop_bad = np.sum(bad_cols) / len(bad_cols)
    vecs_all = vecs_all[:, ~bad_cols]
    vecs_all = stats.rankdata(vecs_all, axis=1)
    RSM_llama = np.corrcoef(vecs_all)
    RSM_llama[np.diag_indices_from(RSM_llama)] = np.nan
    # plt.imshow(RSM_llama)
    # plt.colorbar()
    # plt.show()

    RSM_eat = np.zeros((vecs_all.shape[0], vecs_all.shape[0]))
    RSM_eat[:len(carnivores), :len(carnivores)] = 1
    RSM_eat[len(carnivores):, len(carnivores):] = 1
    # plt.imshow(RSM_eat)
    # plt.show()
    # quit()

    trils = np.tril_indices(RSM_llama.shape[0], -1)

    r, p = stats.spearmanr(RSM_llama[trils], RSM_eat[trils])
    print(f'{layer_name} | llama x eat: {r=:.3f} | {p=:.3f}')


def make_animal_food_vecs(animals, food_match, food_mismatch, d_vecs, odd_even=None,
                          food_second=True):
    X = []
    Y = []
    foods = food_match + food_mismatch
    if odd_even is not None:
        foods = foods[::2] if odd_even == 0 else foods[1::2]
    for animal in animals:
        for food in foods:
            if food_second:
                vecs0 = d_vecs[(animal, food, food)]
            else:
                vecs0 = d_vecs[(food, animal, animal)]
            vecs = vecs0
            X.append(vecs)
            if food in food_match:
                Y.append(1)
            else:
                Y.append(0)
    return np.array(X), np.array(Y)


@pkld(overwrite=False)
def cross_species_regression(layer_name=19, normalize=True,
                             cat='attn_weights',
                             cross_animal=True,
                             activation_model='meta-llama/Llama-3.2-3b',
                             food_second=False
                             ):
    if food_second == 'both':
        kw_outer = {'layer_name': layer_name, 'activation_model': activation_model,
                    'cross_animal': cross_animal, 'food_second': True,
                    'normalize': normalize, 'cat': cat}
        r2_True = cross_species_regression(**kw_outer)
        kw_outer['food_second'] = False
        r2_False = cross_species_regression(**kw_outer)
        return (r2_True + r2_False) / 2
    if cat == 'input' and layer_name == 0 and normalize:
        return np.nan # (all inputs are same so normalize will make all nans)
    # Normalize=True is critical for generalizing from carnivore <-> herbivore
    #   Don't even need odd_even=0/1
    kw = {'layer_name': layer_name,
          'do_unrelated': False,
          'activation_model': activation_model,
          'cat': cat,
          'food_second': food_second
          }

    d_vecs, carnivores, herbivores, meats, plants = (
        pickle_wrap(make_df_carnivore_herbivore,
                    kwargs=kw,
                    easy_override=False, verbose=-1))

    if normalize:
        # keys2 = set(key[2] for key in d_vecs.keys())

        for food in meats + plants:
            for stim_class in [carnivores + herbivores, meats + plants]:
                for i in range(2):
                    vecs = np.array([d_vecs[key] for key in d_vecs.keys() if
                                     key[i] == food and key[2] in stim_class])
                    vecs_M = np.nanmean(vecs, axis=0)
                    vecs_SD = np.nanstd(vecs, axis=0)
                    for key in d_vecs.keys():
                        if key[i] == food and key[2] in stim_class:
                            d_vecs[key] = (d_vecs[key] - vecs_M) / vecs_SD

    animals = carnivores + herbivores
    foods = meats + plants

    X_carn, Y_carn = make_animal_food_vecs(carnivores, meats, plants, d_vecs,
                                           odd_even=0 if cross_animal else None,
                                           food_second=food_second)
    X_herb, Y_herb = make_animal_food_vecs(herbivores, plants, meats, d_vecs,
                                           odd_even=1 if cross_animal else None,
                                           food_second=food_second)

    from sklearn.svm import SVC

    clf = SVC(kernel='linear', C=1)
    X = np.concatenate([X_carn, X_herb], axis=0)
    bad_cols = np.isnan(X).any(axis=0)

    X = X[:, ~bad_cols]
    print(f'Post NaN drop: {X.shape=}')
    bad_cols_inf = np.isinf(X).any(axis=0)
    X = X[:, ~bad_cols_inf]
    print(f'Post inf drop: {X.shape=}')
    y = np.concatenate([Y_carn, Y_herb], axis=0)

    # groups = []
    # for animal in animals:
    #     for food in foods:
    #         groups.append(animal)
    # print(f'{len(groups)=}')
    # print(f'{X.shape=}')
    if cross_animal:
        groups = [0] * X_carn.shape[0] + [1] * X_herb.shape[0]
    else:
        groups = []
        for i, animal in enumerate(carnivores):
            for food in foods:
                groups.append(i)
        for i, animal in enumerate(herbivores):
            for food in foods:
                groups.append(i)


    accs = []
    # for i in range(1 if cross_animal else 10):
    # for i in range(100):
    # cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    # groups = np.arange(len(groups))
    # if cross_animal:
    #     cv = StratifiedGroupKFold(n_splits=2)
    # else:
    cv = LeaveOneGroupOut()

    # cv = SKFold
    # groups = remap_groups()
    accs_cv = []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # print(f'{X_train.shape=}')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = np.mean(y_pred == y_test)
        accs_cv.append(acc)
    acc_cv = np.mean(accs_cv)
    accs.append(acc_cv)
    acc_M = np.mean(accs)
    return acc_M


def plot_layers_cross_species(cross_animal=False,
                              activation_model='meta-llama/Llama-3.3-70b-Instruct',
                              # activation_model='meta-llama/Llama-3.2-3b',
                              food_second=False,
                              plot_title=True, xlabel=True,
                              four_piece=False
                              ):
    vals_attn = []
    vals_residual = []
    vals_attn_output = []
    vals_down = []
    vals_mid = []
    num_layers = 28 if ('3b' in activation_model or '3b' in activation_model[0]) else 80

    for layer_name in range(num_layers):
        layer_name_l = layer_name
        kw = {'layer_name': layer_name_l, 'activation_model': activation_model,
              'cross_animal': cross_animal, 'food_second': food_second}


        if 'bury' not in activation_model:
            r2_attn = cross_species_regression(cat='attn_weights', **kw)
        else:
            r2_attn = np.nan

        r2_gate = cross_species_regression(cat='gate_proj_in', **kw)
        r2_input = cross_species_regression(cat='input', **kw)
        r2_attn_output = cross_species_regression(cat='attn_output', **kw)
        r2_down_proj = cross_species_regression(cat='down_proj_out', **kw)

        print(f'{layer_name} | {r2_attn=:.3f}, {r2_input=:.3f}, {r2_attn_output=:.3f}, {r2_down_proj=:.3f}')

        # if food_second == 'both':
        #     kw = {'layer_name': layer_name_l, 'activation_model': activation_model,
        #       'cross_animal': cross_animal, 'food_second': food_second}
        #     kw['food_second'] = False
        #
        #     if 'bury' not in activation_model:
        #         r2_attn_ = cross_species_regression(cat='attn_weights', **kw)
        #     else:
        #         r2_attn_ = np.nan
        #
        #     r2_gate_ = cross_species_regression(cat='gate_proj_in', **kw)
        #     r2_input_ = cross_species_regression(cat='input', **kw)
        #     r2_attn_output_ = cross_species_regression(cat='attn_output', **kw)
        #     r2_down_proj_ = cross_species_regression(cat='down_proj_out', **kw)
        #
        #     r2_attn = (r2_attn + r2_attn_) / 2
        #     r2_gate = (r2_gate + r2_gate_) / 2
        #     r2_input = (r2_input + r2_input_) / 2
        #     r2_attn_output = (r2_attn_output + r2_attn_output_) / 2
        #     r2_down_proj = (r2_down_proj + r2_down_proj_) / 2
        #
        # print(f'{layer_name} | {r2_attn=:.3f}, {r2_input=:.3f}, {r2_attn_output=:.3f}, {r2_down_proj=:.3f}')

        vals_attn.append(r2_attn)
        vals_residual.append(r2_input)
        vals_attn_output.append(r2_attn_output)
        vals_down.append(r2_down_proj)
        vals_mid.append(r2_gate)

    # plt.plot(list(range(len(vals_attn))), vals_attn,
    #          label='Attention Weights', color='green', marker='.')
    # plt.rcParams.update({'font.size': 14})
    if plot_title:
        plt.title(f'Carn vs. Herb: {food_second=}, {cross_animal=}',
                  fontsize=16)

    vals_l = [vals_residual, vals_attn_output, vals_down,
              # vals_attn
              ]
    labels = ['Residual\n(input)', 'Attention\naddition',
              'FFN\naddition', #'Attention\nweights'
              ]
    colors = ['k', 'r', 'dodgerblue', 'green']
    if four_piece:
        plt.title('Herbivore/carnivore\nx plant/meat')
    else:
        plt.title('Herbivore/carnivore x plant/meat\n"An {animal} and {food}"')
    general_llama_plot(vals_l, labels, colors, ylabel='Accuracy (%)',
                       do_legend=False, num_layers=num_layers,
                       xlabel=xlabel, do_ylabel=not four_piece)
    # plt.show()

if __name__ == '__main__':
    plot_layers_cross_species(cross_animal=True)
    plt.show()
    plot_layers_cross_species(cross_animal=False)
    plt.show()
    # prepare_analogy_figs(activation_model=r'meta-llama/Llama-3.3-70b-Instruct')


