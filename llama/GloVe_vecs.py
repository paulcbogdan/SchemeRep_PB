
import numpy as np
import urllib.request
import zipfile
import os
import numpy as np
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from organize_bhv import get_trial_info
from stim import norm_vectors


def get_glove_vecs_all(dim=300):
    """
    Load GloVe vectors from file or download if not present
    """
    # Define the URL and paths
    # glove_url = f"https://nlp.stanford.edu/data/glove.6B.{dim}d.zip"
    # vector_file = f"glove.6B.{dim}d.txt"
    #
    # # Download and extract if not already present
    # if not os.path.exists(vector_file):
    #     print(f"Downloading GloVe vectors ({dim}d)...")
    #     urllib.request.urlretrieve(glove_url, "glove.zip")
    #     with zipfile.ZipFile("glove.zip", 'r') as zip_ref:
    #         zip_ref.extractall()
    #     os.remove("glove.zip")
    vector_file = r'C:\PycharmProjects\SchemeRep\llama\features\glove.6B\glove.6B.300d.txt'

    # Load vectors into dictionary
    print("Loading GloVe vectors...")
    embeddings_dict = {}
    with open(vector_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector

    return embeddings_dict


def get_glove_vec_deve(word, embeddings_dict):
    """
    Get GloVe vector for a word, handling multi-word items
    """
    # Handle multi-word items
    word = word.replace(' ', '_')
    if '_' in word:
        words = word.split('_')
        # Average the vectors for each word
        vectors = []
        for w in words:
            if w.lower() in embeddings_dict:
                vectors.append(embeddings_dict[w.lower()])
        if vectors:
            return np.mean(vectors, axis=0)
    else:
        if word.lower() in embeddings_dict:
            return embeddings_dict[word.lower()]

    # If word not found, return None or raise exception
    return None

def get_glove_d_vecs_(normalize=True, norm_by_type=True):
    from gensim import downloader
    w2vectors = downloader.load('word2vec-google-news-300')
    print('Loaded word2vec')
    df = get_trial_info('138')
    d_all = {}
    vecs_obj = []
    vecs_scn = []
    embeddings_dict = get_glove_vecs_all()
    renamer = {'haircomb': 'comb', 'soccerball': 'soccer ball',
               'McDonald\'s': 'fast food restuarant'}
    for obj, scene, obj_rename, scene_rename in tqdm(zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']),
                          desc='Getting semantic vectors'):
        if obj_rename in renamer:
            obj_rename = renamer[obj_rename]
        if scene_rename in renamer:
            scene_rename = renamer[scene_rename]

        d_all[obj] = get_glove_vec_deve(obj_rename, embeddings_dict)
        if d_all[obj] is None:
            print(f'Bad obj: {obj}/{obj_rename}')
        d_all[scene] = get_glove_vec_deve(scene_rename, embeddings_dict)
        if d_all[scene] is None:
            print(f'Bad scene: {scene}/{scene_rename}')
        vecs_obj.append(d_all[obj])
        vecs_scn.append(d_all[scene])

    if normalize:
        d_all = norm_vectors(d_all, vecs_obj, vecs_scn, norm_by_type,
                             set(df['obj'].values))

    return d_all

def get_glove_d_vecs(normalize=True):
    d_vecs = pickle_wrap(get_glove_d_vecs_, verbose=-1,
                         kwargs={'normalize': normalize})
    return d_vecs

def get_sn_fp_glove_RSM(sn, fp):
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    d_vecs = get_glove_d_vecs()
    vecs = np.array([d_vecs[item] for item in df_sn['obj']])
    assert np.sum(np.isnan(vecs)) == 0, f'{np.sum(np.isnan(vecs))=}'
    RSM = np.corrcoef(vecs)
    RSM[np.diag_indices_from(RSM)] = np.nan
    return RSM



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    RSM = get_sn_fp_glove_RSM('138', 'obj7_fMRI')
    plt.imshow(RSM)
    plt.show()
    # get_glove_d_vecs()