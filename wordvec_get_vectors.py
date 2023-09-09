from pickle_wrap import pickle_wrap

from prep_names import organize_subj_df
from wordvec_get_bigrams import fit_word2vec
import pickle

def get_vectors():
    parts_do = 25
    fp_model = fr'cache/schemerep_word2vec_model_first_do{parts_do}.pkl'
    model = pickle_wrap(fp_model, lambda: fit_word2vec(parts_do=parts_do),
                        easy_override=False, verbose=True)

    df = organize_subj_df('138')
    d_obj = {}
    d_scene = {}
    d_all = {}
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
        scene = scene.replace(' ', '_')
        d_all[obj] = d_obj[obj] = model.wv[obj]
        d_all[scene] = d_scene[scene] = model.wv[scene]
    return d_all
    #
    # df_as_l = []
    # for key, vec in d_obj.items():
    #     d = {'stim': key, 'vec': vec, 'type': 'obj'}
    #     df_as_l.append(d)
    # for key, vec in d_scene.items():
    #     d = {'stim': key, 'vec': vec, 'type': 'scene'}
    #     df_as_l.append(d)
    #
    # return d_obj, d_scene, df_as_l

if __name__ == '__main__':
    fp_vecs = 'cache/schemerep_sim_vecs.pkl'
    d_obj, d_scene, df_as_l = pickle_wrap(fp_vecs, lambda: get_vectors())


