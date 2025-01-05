from functools import cache
from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.get_obj_scn_vecs import get_llama_extractor, process_cat_cat_inner
from marinate.pkld import pkld


def add_a_an(word):
    if word[0] in 'aeiou':
        return 'an ' + word
    else:
        return 'a ' + word


def make_sentence_analogy(a, b, c, d):
    a = add_a_an(a)
    b = add_a_an(b)
    c = add_a_an(c)
    d = add_a_an(d)

    sentence = f'Like {a} and {b}, {c} and {d}'
    return sentence


def make_sentence_analogy2(*abcd2):
    abcd2 = [add_a_an(x) for x in abcd2]
    sentence = (f'Consider: {abcd2[0]} is to {abcd2[1]}, as {abcd2[2]} is to {abcd2[3]}; '
                f'{abcd2[4]} is to {abcd2[5]}, as {abcd2[6]} is to {abcd2[7]}')
    return sentence


@pkld
def pkld_extract_activations(sentence, target_words,
                             activation_model='meta-llama/Llama-3.2-3b'):
    extractor = get_llama_extractor(model_name=activation_model, )
    t_st = time()
    res = extractor.extract_activations(sentence, target_words)
    print(f'time needed to extract activations (analogy): {time() - t_st:.2f} s')
    return res


@pkld(store='disk')
def extract_abcd(*abcd,
                 activation_model='meta-llama/Llama-3.2-3b',
                 cat='input', layer_name=1):
    abcd = list(abcd)
    if len(abcd) == 4:
        sentence = make_sentence_analogy(*abcd)
    elif len(abcd) == 8:
        sentence = make_sentence_analogy2(*abcd)
    else:
        raise ValueError(f'Bad length: {len(abcd)}')
    res = pkld_extract_activations(sentence, abcd, activation_model)

    d_vecs = {}

    cat_, inner = process_cat_cat_inner(cat)
    for idx_target in range(len(abcd)):
        if cat == 'attn_weights':
            v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                           axis=(0, 1))
        else:
            v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
            if len(v.shape) > 1:
                v = v.reshape(-1)  # reshapes q_proj, k_proj, v_proj, which are (attn_heads, vector)
        d_vecs[tuple(abcd + [idx_target])] = v
    return d_vecs


@pkld(overwrite=True)
def run_analogy_analysis(cat='input', layer_name=1, position=3,
                         activation_model='meta-llama/Llama-3.2-3b',
                         do_r2=False, copies=1, flip_within=True,
                         analogy=1):
    if analogy == 1:
        abcds, relatedness, groups = (
            get_analogy1_abcds(copies=copies, flip_within=flip_within))
    elif isinstance(analogy, tuple) and analogy[0] == 2:
        abcds, relatedness, groups = (
            get_analogy2_abcds(copies=copies, flip_within=flip_within,
                               just1=analogy[1]))
    elif analogy == 2:
        abcds, relatedness, groups = (
            get_analogy2_abcds(copies=copies, flip_within=flip_within))
    else:
        raise ValueError(f'Bad {analogy=}')
    vecs = []
    looper = tqdm(abcds, position=0, leave=True,
                  desc=f'Getting llama: {cat}') if layer_name == 0 else abcds
    if layer_name == 0:
        print(f'{abcds=}')
    for abcd in looper:
        # print(f'{abcd=}')
        d_vecs = extract_abcd(*abcd, cat=cat, layer_name=layer_name,
                              activation_model=activation_model)
        vec = d_vecs[tuple(abcd + [position])]
        vecs.append(vec)
    vecs = np.array(vecs)
    result = fit_regularized_models(vecs, relatedness, plot=False,
                                    n_repeats=1, normalize=False,
                                    groups=groups, do_r2=do_r2)
    r2 = result['Ridge']['r2_score']
    return r2


def compare_GPT_spots70(activation_model='70b', position=7, do_r2=False,
                        copies=1, flip_within=True, analogy=2):
    compare_GPT_spots(activation_model=activation_model, position=position,
                      do_r2=do_r2, copies=copies, flip_within=flip_within,
                      analogy=analogy)

def compare_GPT_spots70_(activation_model='70b', position=1, do_r2=False,
                        copies=0, flip_within=False, analogy=1):
    # TODO: run this as flip_within
    compare_GPT_spots(activation_model=activation_model, position=position,
                      do_r2=do_r2, copies=copies, flip_within=flip_within,
                      analogy=analogy)

# def compare_GPT_spots(activation_model='70b', position=3, do_r2=False,
#                   copies=1, flip_within=True, analogy=1):
def compare_GPT_spots(activation_model='3', position=7, do_r2=False,
                      copies=25, flip_within=True, analogy=(2, 'hard')):#(2, 1)):

    if activation_model == '70b':
        activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    else:
        activation_model = 'meta-llama/Llama-3.2-3b'
    vals_input = []
    vals_down = []
    vals_mid = []
    vals_attn_output = []
    # for layer_name in tqdm(range(28 if '3b' in activation_model else 80),
    #                        desc='Running layers', position=0, leave=True):
    for layer_name in range(28 if '3b' in activation_model else 80):
        kw = {'activation_model': activation_model, 'position': position,
              'do_r2': do_r2, 'copies': copies, 'flip_within': flip_within,
              'layer_name': layer_name, 'analogy': analogy}
        r2_input = run_analogy_analysis(cat='input', **kw)
        r2_down = run_analogy_analysis(cat='down_proj_out', **kw)
        r2_mid = run_analogy_analysis(cat='gate_proj_in', **kw)
        r2_attn_output = run_analogy_analysis(cat='attn_output', **kw)
        vals_input.append(r2_input)
        vals_down.append(r2_down)
        vals_mid.append(r2_mid)
        vals_attn_output.append(r2_attn_output)
        print(f'{layer_name} | {r2_input=:.2f}, {r2_down=:.2f}, '
              f'{r2_mid=:.2f}, {r2_attn_output=:.2f}')

    plt.plot(vals_input, label='Residual (input)',
             color='purple', marker='.', alpha=0.5)
    plt.plot(vals_mid, label='Residual (middle)',
             color='k', marker='.', alpha=0.5)
    plt.plot(vals_attn_output, label='Attention addition',
             color='green', marker='.', alpha=0.5)
    plt.plot(vals_down, label='MLP addition',
             color='blue', marker='.', alpha=0.5)
    plt.title(f'Analogy analysis, {position=}')
    if do_r2:
        plt.ylabel('R^2')
        plt.plot([0, len(vals_input)], [0, 0], color='r', linestyle='--')
    else:
        plt.ylabel('Accuracy')
        plt.plot([0, len(vals_input)], [0.5, 0.5], color='r', linestyle='--')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


@cache
def parse_txts():
    txts = '''Like a glove and a hand, a foot and a shoe
Like a key and a door, a password and a account
Like a pen and a paper, a brush and a canvas
Like a seed and a tree, an egg and a chicken
Like a teacher and a school, a chef and a restaurant
Like a hammer and a nail, a screwdriver and a screw
Like a bird and a nest, a bee and a hive
Like a cake and an oven, a pizza and a stone
Like a wheel and a bicycle, a propeller and a boat
Like a pilot and an airplane, a captain and a ship
Like a cork and a bottle, a plug and a sink
Like a museum and an artifact, a library and a book
Like an anchor and a boat, a root and a tree
Like a feather and a bird, a scale and a fish
Like a paddle and a canoe, a pedal and a bicycle
Like a pillow and a head, a cushon and a back
Like an envelope and a letter, a box and a gift
Like a magnet and a fridge, a hook and a ceiling
Like a camera and a photo, a recorder and a sound
Like a shoe and a footprint, a tire and a track
Like a clock and a wall, a watch and a wrist ------
Like a stove and a pot, a grill and a skewer
Like a farmer and a field, a fisherman and a net
Like a book and a shelf, a file and a folder
Like a bridge and a river, a tunnel and a mountain
Like a violin and a bow, a drum and a stick
Like a key and an ignition, a button and a remote - Copy of key
Like a glove and a hand, a mitten and a finger
Like a leash and a dog, a saddle and a horse
Like a telescope and a star, a microscope and a cell
Like a wheel and a car, a rudder and a boat
Like a phone and a charger, a laptop and a cable
Like a stage and a performer, a podium and a speaker
Like a ladder and a wall, a rope and a cliff
Like a spoon and a soup, a fork and a salad
Like a lock and a chain, a clasp and a bracelet
Like a tree and a swing, a bench and a park
Like a passport and a trip, a ticket and a concert
Like a shadow and a light, an echo and a sound
Like a train and a track, a car and a road
Like a stamp and a letter, a signature and a check
Like a clock and a time, a calendar and a date
Like a fence and a yard, a wall and a room
Like a lock and a chain, a belt and a pants
Like a button and a shirt, a buckle and a belt
Like a stage and a play, a screen and a movie
Like a remote and a TV, a mouse and a screen
Like a torch and a cave, a lamp and a room
Like a comb and a hair, a razor and a beard
Like a nose and a scent, a tongue and a taste'''

    txts = txts.replace(',', '').split('\n')
    abcds = []
    for i, txt in enumerate(txts):
        spl = txt.split(' ')
        try:
            a = spl[2]
            b = spl[5]
            c = spl[7]
            d = spl[10]
        except IndexError as e:
            print(f'{e=}')
            print(f'{txt=}')
            print(f'{spl=}')
            print(f'{a=}, {b=}, {c=}, {d=}')
            quit()
        assert len({a, b, c, d}.intersection({'a', 'an'})) == 0, \
            f'Bad: {txt=}, {spl=}'
        abcds.append([a, b, c, d])
    return abcds


@cache
def get_analogy2_abcds(copies=1, flip_within=True, just1=None):
    abcds_ = parse_txts()
    abcd2s = []
    groups = []
    relatedness = []
    if isinstance(copies, tuple):
        copies, double = copies[0], copies[1]

    else:
        double = False
    for copy in range(copies):
        idxs = np.arange(len(abcds_))
        np.random.seed(copy)
        np.random.shuffle(idxs)
        i_boost = copy * len(idxs)

        for i in range(0, len(abcds_), 2):
            abcd0 = abcds_[idxs[i]]
            abcd1 = abcds_[idxs[i + 1]]
            # if copy == 0:
            #     pass
            # elif copy == 1:
            #     abcd0, abcd1 = abcd1, abcd0
            # else:
            #     raise ValueError(f'Bad number of copies: {copies=}')

            abcd0_flip = flip_abcd(abcd0, flip_within)
            abcd1_flip = flip_abcd(abcd1, flip_within)
            if just1 == 'hard': # test for generalization of gg -> ff
                abcd2_gg = abcd0 + abcd1
                abcd2_ff = abcd0_flip + abcd1_flip
                abcd2_gf = abcd0 + abcd1_flip
                abcd2_fg = abcd0_flip + abcd1
                abcd2s_add = [abcd2_gg, abcd2_ff, abcd2_gf, abcd2_fg]
                relatedness_add = [1, 1, 0, 0]
                # groups.extend([1 + 2 * (i + i_boost),
                #                0 + 2 * (i + i_boost),
                #                0 + 2 * (i + i_boost),
                #                1 + 2 * (i + i_boost)])
                groups.extend([1, 0, 0, 1])
                if double:
                    # groups.extend([1 + 2 * (i + i_boost),
                    #                0 + 2 * (i + i_boost),
                    #                0 + 2 * (i + i_boost),
                    #                1 + 2 * (i + i_boost)])
                    groups.extend([1, 0, 0, 1])
            elif just1 is None:
                abcd2_gg = abcd0 + abcd1
                abcd2_ff = abcd0_flip + abcd1_flip
                abcd2_gf = abcd0 + abcd1_flip
                abcd2_fg = abcd0_flip + abcd1
                abcd2s_add = [abcd2_gg, abcd2_ff, abcd2_gf, abcd2_fg]
                relatedness_add = [1, 1, 0, 0]
            elif just1 == 1:
                abcd2_gg = abcd0 + abcd1
                abcd2_gf = abcd0 + abcd1_flip
                abcd2s_add = [abcd2_gg, abcd2_gf]
                relatedness_add = [1, 0]
            elif just1 == 0:
                abcd2_fg = abcd0_flip + abcd1
                abcd2_gg = abcd0_flip + abcd1_flip
                abcd2s_add = [abcd2_fg, abcd2_gg]
                relatedness_add = [0, 1]
            else:
                raise ValueError

            for abcd2, rel in zip(abcd2s_add, relatedness_add):
                abcd2s.append(abcd2)
                relatedness.append(rel)
                if double:
                    raise ValueError
                    abcd2_valid_alt = [abcd2[1], abcd2[0], abcd2[3], abcd2[2],
                                       abcd2[5], abcd2[4], abcd2[7], abcd2[6]]
                    # abcd2_valid_alt = abcd2[4:] + abcd2[:4]
                    abcd2s.append(abcd2_valid_alt)
                    relatedness.append(rel)
            # abcd2s += abcd2s_add
            if just1 != 'hard':
                groups.extend([i + i_boost] * (len(abcd2s) - len(groups)))
    # print(f'{groups=}')
    # quit()
    relatedness = np.array(relatedness)
    groups = np.array(groups)
    assert len(abcd2s) == len(relatedness) == len(groups)
    return abcd2s, relatedness, groups


def flip_abcd(abcd, flip_within):
    if flip_within:
        return [abcd[0], abcd[1], abcd[3], abcd[2]]
    else:
        return [abcd[0], abcd[3], abcd[2], abcd[1]]


@cache
def get_analogy1_abcds(copies=1, flip_within=True):
    abcds_ = parse_txts()
    abcds = []
    groups = []
    for i, (a, b, c, d) in enumerate(abcds_):
        abcds.append([a, b, c, d])
        copies -= 1
        if copies > 0: abcds.append([b, a, d, c])
        copies -= 1
        if copies > 0: abcds.append([c, d, a, b])
        copies -= 1
        if copies > 0: abcds.append([d, c, b, a])
        groups.extend([i] * (len(abcds) - len(groups)))

    relatedness = [1] * len(abcds)

    abcds_flip = []
    for abcd, group in zip(abcds, groups):
        abcd_flip = flip_abcd(abcd, flip_within)
        abcds_flip.append(abcd_flip)
        groups.append(group)
    relatedness = relatedness + [0] * len(abcds_flip)
    relatedness = np.array(relatedness)

    return abcds + abcds_flip, relatedness, groups


# @pkld(verbose=True)
# def test_pkld2(c, b, **kwargs):
#     return str(c) + str(kwargs['a'])

# Like a bucket and a well, a cup and a tap

if __name__ == '__main__':
    # print(test_pkld2(1, 2, a=3))
    # print(test_pkld2(1, 2, a=3))

    # parse_analogy_txts()
    # compare_GPT_spots()
    compare_GPT_spots70()

