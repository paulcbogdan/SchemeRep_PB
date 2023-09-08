from pickle_wrap import pickle_wrap

from prep_names import organize_subj_df
from wordvec_get_bigrams import fit_word2vec

if __name__ == '__main__':
    # get_bigram_vectors()
    parts_do = 30
    fp_model = fr'cache/schemerep_word2vec_model_first_do{parts_do}.pkl'
    model = pickle_wrap(fp_model, lambda: fit_word2vec(parts_do=parts_do), easy_override=False)

    test = model.wv['the']

    df = organize_subj_df('138')
    d = {}
    for obj, scene in zip(df['obj'], df['scene']):
        # print(obj)
        # obj = scene
        # if obj == 'climbing shoe':
        #     continue
        # if obj == 'ATM':
        #     continue
        # if obj == 'picnic blanket':
        #     continue
        # if obj == 'haircomb':
        #     continue
        # if obj == 'soccer ball':
        #     continue
        # if scene == 'college quad':
        #     continue
        # if scene == 'fast food':
        #     continue
        print(f'1: {obj}')
        obj = obj.replace(' ', '_')
        # scene = scene.replace(' ', '_')
        d[obj] = model.wv[obj]
        # d[scene] = model.wv[scene]
    print(d)



