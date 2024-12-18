from connRSA_finalizing.plot_bars_explore import get_explore_llama, get_explore_BERT


def compare_models(region='ITL'):
    llama2_7b = get_explore_llama(activation_model=r'meta-llama/Llama-2-7b-hf',
                                  attn=False)
    llama31_3b = get_explore_llama(activation_model=r'meta-llama/Llama-3.1-3b',
                                   attn=False)
    llama33_70 = get_explore_llama(activation_model=r'meta-llama/Llama-3.3-70b-Instruct',
                                   attn=False)
    BERT = get_explore_BERT('BERT')
    simCSE = get_explore_BERT('simCSE')
    word2vec = True


if __name__ == '__main__':
    pass
