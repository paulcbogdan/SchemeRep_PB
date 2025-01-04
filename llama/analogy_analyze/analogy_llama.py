from llama.get_obj_scn_vecs import get_llama_extractor
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

@pkld
def pkld_extract_activations(sentence, target_words,
                             activation_model='meta-llama/Llama-3.2-3b'):
    extractor = get_llama_extractor(model_name=activation_model,)
    res = extractor.extract_activations(sentence, target_words)
    return res

def extract_abcd(a, b, c, d,
                 activation_model='meta-llama/Llama-3.2-3b',):
    abcd = [a, b, c, d]
    sentence = make_sentence_analogy(*abcd)
    res = pkld_extract_activations(sentence, abcd, activation_model)
