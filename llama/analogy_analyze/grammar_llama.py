import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.analogy_analyze.analogy_llama import pkld_extract_activations
from llama.get_obj_scn_vecs import process_cat_cat_inner
from marinate.pkld import pkld
from time import time

# suppress: RuntimeWarning: divide by zero encountered in divide
np.seterr(divide='ignore', invalid='ignore')

# I have been running
# I am running
# I was running
# I will be running
# I will have been running
# I had been running

#     Driving
#     Walking
#     Running
#     Jumping
#     Crawling
#     Climbing
#     Swimming
#     Flying
#     Marching
#     Strolling
#     Wandering
#     Paddling
#     Rowing
#     Sailing
#     Hiking
#     Meandering
#     Scramble
#     Scurrying
#     Creeping
#     Riding

#     Reading
#     Writing
#     Knitting
#     Meditating
#     Drawing
#     Painting
#     Thinking
#     Listening
#     Watching
#     Sewing
#     Studying
#     Typing
#     Singing
#     Humming
#     Whistling
#     Reflecting
#     Daydreaming
#     Praying
#     Editing
#     Planning

def get_verbs(verb_type=0):
    movement = ['driving', 'walking', 'running', 'jumping', 'crawling', 'climbing', 'swimming', 'flying', 'marching',
                'strolling', 'wandering', 'paddling', 'rowing', 'sailing', 'hiking', 'meandering', 'scramble',
                'scurrying', 'creeping', 'riding']
    still = ['reading', 'writing', 'knitting', 'meditating', 'drawing', 'painting', 'thinking', 'listening',
             'watching', 'sewing', 'studying', 'typing', 'singing', 'humming', 'whistling', 'reflecting',
             'daydreaming', 'praying', 'editing', 'planning']
    assert len(movement) == len(still) == 20
    verbs = movement + still
    if verb_type == 0:
        content = [0] * len(movement) + [1] * len(still)
        return verbs, content
    elif verb_type == 1:
        activities = [
            'cooking', 'baking', 'gardening', 'cleaning', 'repairing', 'decorating', 'exercising',
            'stretching', 'chatting', 'texting', 'emailing', 'shopping', 'browsing',
            'laughing', 'crying', 'dancing', 'photographing', 'filming', 'playing',
            'gaming', 'learning', 'teaching', 'volunteering', 'hugging', 'kissing', 'sleeping',
            'napping', 'dreaming', 'snacking', 'eating', 'drinking', 'sipping', 'smoking', 'vaping',
            'crafting', 'woodworking', 'pottery-making', 'jewelry-making', 'crocheting',
            'embroidering', 'quilting', 'scrapbooking', 'blogging', 'podcasting', 'vlogging',
            'streaming', 'fishing', 'birdwatching', 'stargazing', 'journaling',
            'doodling', 'coloring', 'puzzle-solving', 'sudoku-playing', 'crossword-doing',
            'brainstorming', 'inventing', 'composing', 'sketching', 'sculpting', 'mowing', 'weeding',
            'harvesting', 'canning', 'brewing', 'fermenting', 'composting',
            'debating', 'discussing', 'negotiating', 'advising', 'mentoring',
            'counseling', 'healing', 'massaging', 'manicuring', 'pedicuring', 'stylizing', 'barbering',
            'makeup-applying', 'tattooing', 'piercing', 'diagnosing', 'therapizing',
            'coaching', 'training', 'cheering', 'celebrating', 'mourning', 'remembering', 'forgiving',
            'loving'
        ]
        activities = activities[:88]
        verbs = verbs + activities
        overlap_any = set(movement) & set(activities)
        assert len(overlap_any) == 0, overlap_any
        overlap_any = set(still) & set(activities)
        assert len(overlap_any) == 0, overlap_any
        assert len(verbs) == len(set(verbs))
        content = [0] * len(movement) + [1] * len(still) + [2] * len(activities)
        return verbs, content


def get_verb_sentences(verb, not_str=''):
    sentence0 = f"I am{not_str} {verb}"
    sentence1 = f"I was{not_str} {verb}"
    sentence2 = f"I have{not_str} been {verb}"
    sentence3 = f"I had{not_str} been {verb}"
    sentence4 = f"I will{not_str} be {verb}"
    sentence5 = f"I will{not_str} have been {verb}"

    sentence6 = f"They are{not_str} {verb}"
    sentence7 = f"They were{not_str} {verb}"
    sentence8 = f"They have{not_str} been {verb}"
    sentence9 = f"They had{not_str} been {verb}"
    sentence10 = f"They will{not_str} be {verb}"
    sentence11 = f"They will{not_str} have been {verb}"

    d = {'am': sentence0, 'was': sentence1, 'have been': sentence2,
         'had been': sentence3, 'will be': sentence4, 'will have been': sentence5,
         'they are': sentence6, 'they were': sentence7,
         'they have been': sentence8, 'they had been': sentence9,
         'they will be': sentence10, 'they will have been': sentence11}
    order = ['am', 'was', 'have been', 'had been', 'will be', 'will have been',
             'they are', 'they were', 'they have been', 'they had been',
             'they will be', 'they will have been']

    if not_str == ' not':
        d_ = {}
        for key, sentence in d.items():
            d_[key + not_str] = sentence
        d = d_
        order = [key + not_str for key in order]
    else:
        d_not, order_not = get_verb_sentences(verb, not_str=' not')
        d.update(d_not)
        order.extend(order_not)

    return d, order


def get_vecs_verb(verb, activation_model='meta-llama/Llama-3.2-3b',
                  cat='input', layer_name=1, just_do=None):

    if isinstance(activation_model, tuple):
        if activation_model[1] == 'bury':
            add_on = (f'I thought about this for a long while. '
                      f'The more I pondered, the clearer it became '
                      f'that my initial reaction was just the tip of '
                      f'the iceberg. There were layers to this issue, '
                      f'complexities that I hadn\'t considered at '
                      f'first glance. Each new angle brought a '
                      f'different perspective, challenging my '
                      f'assumptions and making me question what I '
                      f'thought I knew. It was like peeling an onion, '
                      f'revealing not just answers, but more questions, '
                      f'more nuances to explore')
            # activation_model = activation_model[0]
        else:
            raise ValueError
    d, order = get_verb_sentences(verb)
    cat_, inner = process_cat_cat_inner(cat)
    d_vecs = {}
    for key in order:
        if just_do is not None:
            if key not in just_do:
                continue
        sentence = d[key]
        if isinstance(activation_model, tuple):
            words = add_on.split(' ')
            if activation_model[1] == 'bury':
                sentence = f'{sentence}. {add_on}'
            else:
                raise ValueError

            activation_model = activation_model[0]
            res = pkld_extract_activations(sentence, [words[-1]], activation_model)
        else:
            res = pkld_extract_activations(sentence, [verb], activation_model)
        if cat == 'attn_weights':
            v = np.nanmean(res['attn']['attn_weights'][layer_name][0],
                           axis=(0, 1))
        else:
            v = np.nanmean(res[inner][cat_][layer_name][0], axis=0)
            if len(v.shape) > 1:
                v = v.reshape(-1)
        d_vecs[(key, verb)] = v

    return d_vecs


def get_vecs_verbs(verb_types=0, activation_model='meta-llama/Llama-3.2-3b',
                   cat='input', layer_name=1, just_do=None):
    verbs, content = get_verbs(verb_types)
    d_vecs_all = {}
    for verb in verbs:
        d_vecs = get_vecs_verb(verb, activation_model, cat, layer_name,
                               just_do=just_do)
        d_vecs_all.update(d_vecs)
    return d_vecs_all


@pkld
def run_verb_analysis(cat='input', layer_name=1, activation_model='meta-llama/Llama-3.2-3b',
                      verb_types=0, comparison=None, normalize=False, groupby=False,
                      do_r2='discrete',
                      reverse=True):
    if isinstance(comparison[0], tuple) or isinstance(comparison[0], list):
        just_do = []
        for comp in comparison:
            just_do.extend(comp)
    else:
        just_do = None if comparison == 'content' else comparison
    print(f'{just_do=}')
    # print(activation_model)
    # quit()
    d_vecs = get_vecs_verbs(verb_types, activation_model, cat, layer_name,
                            just_do=just_do)
    verbs, content = get_verbs(verb_types)
    if comparison == 'content':
        groups = []
        forms = list(set(key[0] for key in d_vecs))
        vecs_all = []
        y = []
        for verb, verb_content in zip(verbs, content):
            verb_vecs = []
            for form in forms:
                vec = d_vecs[(form, verb)]
                verb_vecs.append(vec)
                y.append(verb_content)
                groups.append(form)
            if normalize:
                verb_vecs = np.array(verb_vecs)
                verb_vecs = stats.zscore(verb_vecs, axis=0, nan_policy='omit')
                vecs_all.extend(list(verb_vecs))
            else:
                vecs_all.extend(verb_vecs)
    else:
        groups = []
        vecs_all = []
        y = []
        for verb in verbs:
            verb_vecs = []
            for comp_i, comp in enumerate(comparison):
                if isinstance(comp, list) or isinstance(comp, tuple):
                    for j, subcomp in enumerate(comp):
                        vec = d_vecs[(subcomp, verb)]
                        verb_vecs.append(vec)
                        if reverse: # compare across comp and generalize between comp
                            y.append(j)
                            groups.append(comp_i)
                        else:
                            y.append(comp_i)
                            groups.append(j)
                        # print(f'{subcomp=}, {comp_i=}, {j=}')
                else:
                    vec = d_vecs[(comp, verb)]
                    verb_vecs.append(vec)
                    y.append(comp_i)
                    groups.append(verb)
            if normalize:
                verb_vecs = np.array(verb_vecs)
                # print(f'{verb_vecs=}')
                verb_vecs = stats.zscore(verb_vecs, axis=0, nan_policy='omit')
                vecs_all.extend(list(verb_vecs))
                # print(f'{verb_vecs=}')
                # quit()
            else:
                vecs_all.extend(verb_vecs)
    # print(f'{groups=}')
    # print(f'{y=}')
    # quit()
    assert len(np.unique(groups)) == 2
    vecs_all = np.array(vecs_all)
    y = np.array(y)
    bad_cols = np.isnan(vecs_all).any(axis=0)
    vecs_all = vecs_all[:, ~bad_cols]
    inf_cols = np.isinf(vecs_all).any(axis=0)
    vecs_all = vecs_all[:, ~inf_cols]
    if vecs_all.shape[1] < 10:
        # print(f'Not enough data: {vecs_all.shape[1]}')
        # quit()
        return {'Ridge': {'r2_score': np.nan}}
    result = fit_regularized_models(vecs_all, y, plot=False,
                                    n_repeats=1, normalize=False,
                                    groups=groups if groupby else None,
                                    do_r2=do_r2)
    r2 = result['Ridge']['r2_score']
    return r2


@pkld
def compare_GPT_grammar(activation_model='3', do_r2='discrete',
                        comparison=('am', 'was'), groupby=True,
                        normalize=False, verb_types=1,
                        reverse=True):
    if isinstance(activation_model, tuple):
        pass
    elif activation_model == '70b':
        activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
        verb_types = 0
    else:
        activation_model = 'meta-llama/Llama-3.2-3b'
    # print(activation_model)
    # quit()
    vals_input = []
    vals_down = []
    vals_mid = []
    vals_attn_output = []
    # for layer_name in tqdm(range(28 if '3b' in activation_model else 80),
    #                        desc='Running layers', position=0, leave=True):
    for layer_name in range(0, 28 if ('3b' in activation_model or
                                      '3b' in activation_model[0]) else 80):
        kw = {'activation_model': activation_model, 'do_r2': do_r2,
              'layer_name': layer_name, 'comparison': comparison,
              'groupby': groupby, 'normalize': normalize,
              'verb_types': verb_types, 'reverse': reverse}
        r2_input = run_verb_analysis(cat='input', **kw)
        r2_down = run_verb_analysis(cat='down_proj_out', **kw)
        r2_mid = run_verb_analysis(cat='gate_proj_in', **kw)
        r2_attn_output = run_verb_analysis(cat='attn_output', **kw)
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
    plt.title(f'Comparison: {comparison}')
    if do_r2 and do_r2 != 'discrete':
        plt.ylabel('R^2')
        plt.plot([0, len(vals_input)], [0, 0], color='r', linestyle='--')
    else:
        plt.ylabel('Accuracy')
        plt.plot([0, len(vals_input)], [0.5, 0.5], color='r', linestyle='--')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    return vals_input, vals_down, vals_mid, vals_attn_output


def compare_GPT_grammar_all_comparisons(activation_model='3b',
                                        do_r2='discrete', add_they=False,
                                        add_not=False):
    # add_not = False
    add_they = False
    # activation_model = ('meta-llama/Llama-3.2-3b', 'bury')
    forms = [('am', 'was'), ('have been', 'had been'),
             # ('will be', 'will have been'),
    #          # ('will have been', 'will be', ),
             ]

    if add_they:
        forms_they = [('they are', 'they were'), ('they have been', 'they had been'),
                      # ('they will be', 'they will have been')
                      ]
        forms.extend(forms_they)
    if add_not:
        forms_not = [('am not', 'was not'), ('have been not', 'had been not'),
                     # ('will be not', 'will have been not')
                     # ('will have been not', 'will be not',)
                     ]
        if add_they:
            forms_they_not = [('they are not', 'they were not'),
                              # ('they have been not', 'they had been not'),
                              # ('they will be not', 'they will have been not')
                              ]
            forms_not.extend(forms_they_not)
        forms.extend(forms_not)

    # forms = [('am', 'was'), ('they are', 'they were')]

    vals_input, vals_down, vals_mid, vals_attn_output = [], [], [], []
    for i, form0 in enumerate(forms):
        for j, form1 in enumerate(forms):
            # is_good0 = form0[0] in form1[0]
            # is_good1 = form0[1] in form1[1]
            # print(f'{form0}, {form1} | {is_good0=}, {is_good1=}')
            # if not is_good0 or not is_good1:
            #     continue
            # if form0 == ('am', 'was'):
            #     form0 = ('was', 'am')
            # is_will0 = 'will' in form0[0]
            # is_will1 = 'will' in form1[0]
            # print(f'{form0=}, {form1=} | {is_will0=}, {is_will1=}')
            # if not is_will0 and not is_will1:
            #     continue
            # if is_will0 == is_will1:
            #     continue
            if i >= j:
                continue
            # if i >= j:
            #     continue
            # if 'not' in form0[0]:
            #     continue
            # if not ('not' in form1[0]):
            #     continue

            comparison = (form0, form1)
            t_st = time()
            vals_input_, vals_down_, vals_mid_, vals_attn_output_ = (
                compare_GPT_grammar(comparison=comparison,
                                    activation_model=activation_model,
                                    do_r2=do_r2, reverse=False))
            print(f'Time needed to do grammar comparison: {time() - t_st:.2f} s')
            if np.nanmean(np.array(vals_input_)[5:15]) > 0.98:
                print(f'skip: {form0}/{form1}')
                continue
            print(f'Do: {form0}/{form1} | '
                  f'{np.nanmean(np.array(vals_input_)[5:15])=}')
            vals_input.append(vals_input_)
            vals_down.append(vals_down_)
            vals_mid.append(vals_mid_)
            vals_attn_output.append(vals_attn_output_)
    vals_input = np.array(vals_input)
    vals_input = np.nanmean(vals_input, axis=0)
    vals_down = np.array(vals_down)
    vals_down = np.nanmean(vals_down, axis=0)
    vals_mid = np.array(vals_mid)
    vals_mid = np.nanmean(vals_mid, axis=0)
    vals_attn_output = np.array(vals_attn_output)
    vals_attn_output = np.nanmean(vals_attn_output, axis=0)

    plt.plot(vals_input, label='Residual (input)',
             color='purple', marker='.', alpha=0.5)
    plt.plot(vals_mid, label='Residual (middle)',
             color='k', marker='.', alpha=0.5)
    plt.plot(vals_attn_output, label='Attention addition',
             color='green', marker='.', alpha=0.5)
    plt.plot(vals_down, label='MLP addition',
             color='blue', marker='.', alpha=0.5)
    plt.title(f'Comparison: all')
    print(f'{vals_input=}')
    if do_r2 and do_r2 != 'discrete':
        plt.ylabel('R^2')
        plt.plot([0, len(vals_input)], [0, 0], color='r', linestyle='--')
    else:
        plt.ylabel('Accuracy')
        plt.plot([0, len(vals_input)], [0.5, 0.5], color='r', linestyle='--')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def make_attention_oscillation():
    forms = [('am', 'was'), ('have been', 'had been'),
             # ('will be', 'will have been'),
             # ('will have been', 'will be', ),
             ]
    comparison = (form0, form1)
    t_st = time()
    vals_input_, vals_down_, vals_mid_, vals_attn_output_ = (
        compare_GPT_grammar(comparison=comparison,
                            activation_model=activation_model,
                            do_r2=do_r2))


if __name__ == '__main__':
    # compare_GPT_grammar_all_comparisons()
    # compare_GPT_grammar_all_comparisons(add_they=True, add_not=False)
    compare_GPT_grammar_all_comparisons(add_they=True, add_not=True)

