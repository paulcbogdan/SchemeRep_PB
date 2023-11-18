import gensim.downloader as api

from organize_bhv import get_trial_info


# wv = api.load('word2vec-google-news-300')
# vec_king = wv['goggles']
# print(f'{vec_king=}')


if __name__ == '__main__':
    # get_word2vecs()
    word = 'ferris_wheel'

    from gensim.models import Phrases
    from gensim.test.utils import common_texts
    from gensim.models import Word2Vec
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

    bigram_transformer = Phrases(common_texts, connector_words=ENGLISH_CONNECTOR_WORDS)
    # print(len(common_texts))
    # print(common_texts)
    print(bigram_transformer[common_texts][1])
    # print(common_texts[2])
    quit()
    model = Word2Vec(bigram_transformer[common_texts], min_count=1,
                     vector_size=10, window=5, workers=4)
    # model.save("w2v_test.model")
    test = model.train([['ferris_wheel', 'bike', 'car', 'train', 'roller_coaster',
                  'pier']], total_examples=1, epochs=1)
    print(test)
    sims = model.wv.most_similar('ferris_wheel', topn=4)
    print(sims)


