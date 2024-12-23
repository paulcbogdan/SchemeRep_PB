import pickle

import numpy as np
from torch.nn.init import zeros_
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve, get_standard_items_list
from llama.get_obj_scn_vecs import process_cat_cat_inner
import scipy.stats as stats
import matplotlib.pyplot as plt


def make_df_carnivore_herbivore(activation_model='meta-llama/Llama-3.2-3b',
                                cat='gate_proj_in', layer_name=1,
                                do_unrelated=False):
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
                item0 = animal if flip else food
                item1 = food if flip else animal
                res, fp = pickle_wrap(get_llama_activations_deve,
                                      kwargs={'item0': item0, 'item1': item1,
                                              'activation_model': activation_model},
                                      easy_override=False, verbose=-1, dir_branches=1000,
                                      RAM_cache=True, get_fp=True)

                if 'mlp_out' in res:
                    del res['mlp_out']
                    del res['mlp_in']['down_proj']
                    del res['mlp_in']['up_proj']
                    del res['mlp_in']['act_fn']
                    del res['attn']['q_proj']
                    del res['attn']['k_proj']

                    with open(fp, 'wb') as f:
                        pickle.dump(res, f)

                if isinstance(res, dict):
                    for inner_, d in res.items():
                        if inner_ in ['mlp_out', 'mlp_in', 'attn']:
                            del_keys = []
                            for outer in d.keys():
                                if outer != cat_:
                                    del_keys.append(outer)
                            for outer in del_keys:
                                del res[inner_][outer]

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
                        # if idx_target == 0:
                            # print(f'({item0}, {item1}) | {idx_target}: {v[:4]=}')
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
    # meats_ = meats[::2] + plants[1::2]
    # plants_ = meats[1::2] + plants[::2]
    # meats = meats_
    # plants = plants_
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
                vecs_all.append(d_vecs[(food, animal, food)])# - animal_M)
            else:
                vecs_all.append(d_vecs[(animal, food, food)])# - animal_M)
        # vecs_all.append(vecs)
    vecs_all = np.array(vecs_all)
    bad_cols = np.isnan(vecs_all).any(axis=0)
    prop_bad = np.sum(bad_cols) / len(bad_cols)
    # print(f'Proportion of bad columns: {prop_bad:.3%}')
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


    vecs_carn = get_animals_vecs(carnivores, meats, # + plants, #  meats
                                 d_vecs, reverse=reverse,
                                 get_food=get_food)
    # vecs_carn_r = get_animals_vecs(carnivores, plants,# + meats, # + plants, #  meats
    #                                 d_vecs, reverse=reverse,
    #                                 get_food=get_food)
    # vecs_carn = vecs_carn - vecs_carn_r
    vecs_herb = get_animals_vecs(herbivores, meats,  # plants
                                 d_vecs, reverse=reverse,
                                 get_food=get_food)
    # vecs_herb_r = get_animals_vecs(herbivores, plants,  # plants
    #                                d_vecs, reverse=reverse,
    #                                get_food=get_food)
    # vecs_herb = vecs_herb - vecs_herb_r

    # vecs_carn = get_animals_vecs(carnivores, meats, #  meats
    #                              d_vecs, reverse=reverse,
    #                              get_food=get_food)
    # vecs_herb = get_animals_vecs(herbivores, plants, # plants
    #                              d_vecs, reverse=reverse,
    #                              get_food=get_food)

    # vecs_carn = get_animals_vecs(meats, herbivores + carnivores,
    #                              d_vecs, reverse=reverse,
    #                              get_food=get_food)
    # vecs_herb = get_animals_vecs(plants, herbivores + carnivores,
    #                              d_vecs, reverse=reverse,
    #                              get_food=get_food)
    vecs_all = np.concatenate([vecs_carn, vecs_herb], axis=0)
    bad_cols = np.isnan(vecs_all).any(axis=0)
    prop_bad = np.sum(bad_cols) / len(bad_cols)
    # print(f'Proportion of bad columns: {prop_bad:.3%}')
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
    # quit()

if __name__ == '__main__':
    for layer_name in range(0, 28):
        do_animal_food_animal_food(layer_name)
        # do_carnivore_herbivore(layer_name)
