import gensim.downloader as api

from test2 import organize_subj_df


# wv = api.load('word2vec-google-news-300')
# vec_king = wv['goggles']
# print(f'{vec_king=}')

def get_word2vecs():
    df = organize_subj_df('138')
    wv = api.load('word2vec-google-news-300')
    obj2vec = {}
    scene2vec = {}
    for (obj, scene) in zip(df['obj'], df['scene']):
        try:
            obj2vec[obj] = wv[obj]
            print('IN')
        except:
            try:
                obj2vec[obj] = wv[obj.replace(' ', '')]
            except:
                print(f'Object {obj} not in wv')

        try:
            scene2vec[scene] = wv[scene]
        except:
            try:
                scene2vec[scene] = wv[scene.replace(' ', '')]
            except:
                print(f'Scene {scene} not in wv')

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


