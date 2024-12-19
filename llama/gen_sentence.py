# from accelerate.test_utils.scripts.test_distributed_data_loop import BATCH_SIZE
from tqdm import tqdm
from functools import cache
from time import time

import pandas as pd
import numpy as np

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.gen_sentence_grok import grok_prompt
from organize_bhv import get_trial_info


def generate_sentences_multi():
    pass

def generate_sentences_API(words, num_sentences=8, max_words=25, seed=0,
                           extra_instructions='', be_concise=True,
                           easy_override=False, redo=False):

    prompt = (f"Generate {num_sentences} sentences "
              f"that include both '{words[0]}' and "
              f"'{words[1]}' in a realistic context. "
              f"Be concise (under {max_words} words). "
              f"Do not include any further text. "
              f"Separate sentences with linebreaks "
              f"and numbers (1., 2., etc).")
    # if 'coffee machine' not in words:
    #     return

    # if 'climbing wall' in words:
    #     seed += 10

    if redo:
        prompt = (f"Generate {num_sentences} sentences "
                  f"that include both '{words[0]}' and "
                  f"'{words[1]}' in a realistic context; "
                  f"include those nouns exactly."
                  f"Be concise (under {max_words} words). "
                  f"Do not include any further text. "
                  f"Separate sentences with linebreaks "
                  f"and numbers (1., 2., etc).")
    else:
        prompt = (f"Generate {num_sentences} sentences "
                  f"that include both '{words[0]}' and "
                  f"'{words[1]}' in a realistic context; "
                  f"include those terms exactly."
                  f"Be concise (under {max_words} words). "
                  f"Do not include any further text. "
                  f"Separate sentences with linebreaks "
                  f"and numbers (1., 2., etc).")

    max_output_tokens = max_words * num_sentences * 2
    kw = {'prompt': prompt, 'max_tokens': max_output_tokens,
          'seed': seed}

    result = pickle_wrap(grok_prompt, kwargs=kw,
                         easy_override=easy_override
                         # easy_override='ice skate' in words
                         )
    response = result['response']
    sentences = response.split('\n')
    all_good = True
    print(f'Words: {words}')
    sentences_ = []
    print(sentences)
    for i, sentence in enumerate(sentences):
        # assert words[0] in sentence, f'{words[0].lower()} not in: {sentence=}'
        # assert words[1] in sentence, f'{words[1]} not in: {sentence=}'
        print(f'{sentence}')
        i_str = f'{i + 1}. '
        if not sentence.startswith(i_str):
            all_good = False
            raise ValueError(f'Bad sentence ({words}; {i_str=}): {sentence=}')
        if 'prison' in words:
            sentence = sentence.replace('prisoner', 'man')
        good0 = word_in_sentence_check(words[0], sentence)
        good1 = word_in_sentence_check(words[1], sentence)
        sentence = sentence.replace(i_str, '')
        # print((words[1] + 's') in sentence)
        # print(f'{words=}')
        # print(('football' in words))
        # print(('football field' in words))
        # print(f'{good0=}')
        # print(f'{good1=}')
        if ('football' in words) and ('football field' in words):
            if (not good0) or (not good1):
                print('skip')
                pass
            else:
                sentence_sans_field = sentence.replace('football field', '')
                print(f'{sentence_sans_field=}')
                if 'football' in sentence_sans_field:
                    pass
                else:
                    good0 = False

        if good0 and good1:
            sentences_.append(sentence)
        else:
            continue
            # if True:
            #     pass
                # quit()
            # if set(words).intersection({'coffee machine',
            #                             'gas can',
            #                             'climbing wall', 'swimming pool',
            #                             'dump'
            #                             # 'bar'
            #                             # 'popcorn machine'
            #                             }):
            #     for _ in range(5):
            #         print(f'Bad sentence ({words}): {sentence=}')
            #         print('Redoing!')
            #     return generate_sentences_API(words, num_sentences=num_sentences,
            #                                     max_words=max_words, seed=seed,
            #                                     extra_instructions=extra_instructions,
            #                                     be_concise=be_concise,
            #                                     redo=True, easy_override=redo)
            # else:
            #     raise ValueError(f'Bad sentence ({words}): {sentence=}')
    if len(sentences_) < num_sentences:
        num_left_needed = num_sentences - len(sentences_)
        extra_sentences = generate_sentences_API(words, num_sentences=num_left_needed,
                                                max_words=max_words, seed=seed + 1,
                                                extra_instructions=extra_instructions,
                                                be_concise=be_concise,
                                                redo=True, easy_override=easy_override)
        sentences_ += extra_sentences
    assert len(sentences_) == num_sentences

    assert len(sentences_)
    return sentences_



def word_in_sentence_check(word, sentence):
    # print(word)
    if ((f'{word}ing' in sentence) or (f'{word}er' in sentence) or (f'{word}d' in sentence)\
            or (f'{word}ed' in sentence) or (f'{word}ly' in sentence) or
            (f'{word}al' in sentence)):
        return False
    # if ((word in sentence) and (f' {word} ' not in sentence) and
    #         (f'{word.capitalize()} ' not in sentence) and (f' {word}.' not in sentence) and
    #         (f' {word},' not in sentence)):
    #     print(f'FAIL 2 ({word}): {sentence=}')
    #     quit()
    #     return False
    if word == 'bar':
        if ' bar ' in sentence:
            return True
        if 'bartender' in sentence:
            return False
    if word == 'dump':
        if 'dumped' in sentence:
            return False
    if (word.capitalize() in sentence or word.lower() in sentence or word in sentence or
            (f'{word}s' in sentence) or f'{word}es' in sentence or f'{word}\'s' in sentence):
        return True
    elif ((word.replace('y', 'ies').capitalize() in sentence) or
          (word.replace('y', 'ies').lower() in sentence)):
        return True
    elif word.replace('us', 'i') in sentence:
        return True
    else:
        return False

    word_in_capitalized = f'{word.capitalize()} ' in sentence
    word_in_capitalized_plural = f'{word.capitalize()}s ' in sentence
    word_in_capitalized_plural_e = f'{word.capitalize()}es ' in sentence

    word_in_space_before = f' {word.lower()}' in sentence.lower()
    word_in_space_after = f'{word.lower()} ' in sentence.lower()
    word_in_space_after_plural = (f'{word.lower()}s ' in sentence.lower() or
                                  f'{word.lower()}\'s ' in sentence.lower())
    word_in_space_after_plural_e = f'{word.lower()}es ' in sentence.lower()

    word_in_period = (f'{word.lower()}.' in sentence.lower() or
                      f'{word.lower()},' in sentence.lower() or
                      f'{word.lower()}-' in sentence.lower())
    word_in_period_plural = (f'{word.lower()}s.' in sentence.lower() or
                             f'{word.lower()}s,' in sentence.lower() or
                             f'{word.lower()}\'s,' in sentence.lower())
    word_in_period_plural_e = (f'{word.lower()}es.' in sentence.lower() or
                               f'{word.lower()}es,' in sentence.lower())
    if (word_in_capitalized or word_in_capitalized_plural or
        word_in_capitalized_plural_e) and not word_in_period:
        word_in_strict = True
    elif word_in_space_before and (word_in_space_after or word_in_period or
                                   word_in_space_after_plural or
                                   word_in_period_plural or
                                   word_in_space_after_plural_e or
                                   word_in_period_plural_e):
        word_in_strict = True
    else:
        word_in_strict = False

    return word_in_strict


def get_all_word_pairs():
    # after sn = 111, I think I properly standardized everything
    # sns = list(range(111, 124))
    sns = list(range(102, 124))

    all_pairs = set()
    for sn in sns:
        fp_processed = fr'Behavioral\{sn}\bhv_data_p{sn}.csv'
        df = pd.read_csv(fp_processed)
        df_encoding = df[df['category'] == 'encoding']
        encoding_pairs = df_encoding[['item1', 'item2']].apply(
            tuple, axis=1).apply(lambda x: tuple(sorted(x))).to_list()
        all_pairs.update(encoding_pairs)
    all_pairs = list(all_pairs)
    all_pairs.sort()
    return all_pairs


def make_all_pairs_sentences_API(all_possible=False):
    sns = ['102', '103', '104'] # everyone else is a duplicate
    already_done = set()
    scn_objs = []
    obj_scns = []
    d_vecs = {}
    for sn in sns:
        df_sn = get_trial_info(sn)
        objs = df_sn['obj'].to_list()
        scns = df_sn['scene'].to_list()
        if all_possible:
            objs_ = []
            scns_ = []
            for obj in objs:
                for scn in scns:
                    objs_.append(obj)
                    scns_.append(scn)
            objs = objs_
            scns = scns_

        for obj, scn in zip(objs, scns):
            if (obj, scn) in already_done: continue
            already_done.add((obj, scn))
            scn_objs.append((scn, obj))
            obj_scns.append((obj, scn))
            print(f'Cooking sentences: {obj}/{scn}')
            # if 'prison' not in scn:
            #     continue

            sentences = pickle_wrap(generate_sentences_API,
                                    kwargs={'words': [scn, obj]},
                                    easy_override=True)

if __name__ == "__main__":
    make_all_pairs_sentences_API()