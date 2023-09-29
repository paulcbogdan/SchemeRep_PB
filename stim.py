from pathlib import Path

import gensim
import numpy as np
import pandas as pd
from PIL import Image
from pickle_wrap import pickle_wrap
from sklearn import decomposition
from torchvision import models as models, transforms as transforms

from organize_bhv import get_trial_info


def get_stim_RDM(df_sn, d_vecs, obj_only=False, scene_only=False,
                 dif=False, prod=False, take_abs=False, add=False):
    # return get_stim_RDM_lifu(df_sn)
    assert obj_only or scene_only or dif or prod or add, \
        'Must specify one of obj_only, scene_only, dif, prod, add'
    vec_size = len(d_vecs[df_sn['obj'].iloc[0]])
    vecs_obj = np.empty((len(df_sn['obj']), vec_size))
    vecs_scene = np.empty((len(df_sn['obj']), vec_size))
    for i, (obj, scene, obj_rename, scene_rename) in enumerate(zip(df_sn['obj'],
            df_sn['scene'], df_sn['obj_rename'], df_sn['scene_rename'])):
        vec_obj = d_vecs[obj]
        vec_scene = d_vecs[scene]
        vecs_obj[i, :] = vec_obj
        vecs_scene[i, :] = vec_scene
    if obj_only:
        all_vecs = vecs_obj
    elif scene_only:
        all_vecs = vecs_scene
    elif dif:
        all_vecs = vecs_obj - vecs_scene
    elif prod:
        all_vecs = vecs_obj * vecs_scene
    elif add:
        all_vecs = vecs_obj + vecs_scene
    else:
        raise ValueError('How?? This error should\'ve been caught by assert.')
    all_vecs = np.abs(all_vecs) if take_abs else all_vecs
    RDM_stim = np.corrcoef(all_vecs)
    return RDM_stim


def get_vec(stim, w2v):
    parts = stim.split(' ')
    vecs = []
    for part in parts:
        try:
            vec = w2v[part]
            vecs.append(vec)
        except KeyError:
            print(f'Bad {stim}: {part}')
    return np.mean(vecs, axis=0)


def get_semantic_vectors_():
    w2vectors = gensim.downloader.load('word2vec-google-news-300')
    print('Loaded word2vec')
    df = get_trial_info('138')
    d_all = {}
    for obj, scene, obj_rename, scene_rename in zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']):
        d_all[obj] = get_vec(obj_rename, w2vectors)
        d_all[scene] = get_vec(scene_rename, w2vectors)
    return d_all


def get_semantic_vectors():
    # fit using python 3.11
    fp_vecs = f'cache/schemerep_sem_vecs.pkl'
    d_vecs = pickle_wrap(fp_vecs, get_semantic_vectors_,
                         easy_override=False)
    return d_vecs


def get_DNN_vecs_(PCA=False, DNN_layer=2):
    model = models.vgg16(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    data_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])

    names, fps = get_img_fns()
    img_vecs = []
    obj_vecs = []
    names_sanity = []
    for i, (name, fp) in enumerate(zip(names, fps)):
        if 'Scene' in fp:
            scene = 1
        else:
            scene = 0
        img = Image.open(fp)
        img = np.array(img, dtype=np.uint8)
        x = data_transforms(img).unsqueeze(0)
        if DNN_layer == -1:
            x = model.forward(x)
        else:
            for j in range(DNN_layer):
                x = model.features[j](x)
        # if early:
        #     x = model.features[0](x)
        #     x = model.features[1](x)
        # else:
        #     x = model.forward(x)
        x = x.detach().numpy().flatten()
        x = np.concatenate([x, np.array([scene])])
        img_vecs.append(x)
        if not scene:
            obj_vecs.append(x)
        names_sanity.append(name)

    img_vecs = np.array(img_vecs)
    obj_vecs = np.array(obj_vecs)

    if PCA:
        pca = decomposition.PCA()
        pca.fit(obj_vecs)
        img_vec_brief = pca.transform(img_vecs)
        img_vec_brief = img_vec_brief
    else:
        img_vec_brief = img_vecs

    d_vecs = {}
    for name, vec in zip(names_sanity, img_vec_brief):
        d_vecs[name] = vec
    return d_vecs


def get_DNN_vecs(PCA=False, DNN_layer=2):
    dnn_str = 'late' if DNN_layer == -1 else \
              'early' if DNN_layer == 2 else \
              f'dnn{DNN_layer}'
    PCA_str = '_PCA' if PCA else ''
    fp_DNN_vecs = fr'cache/DNN_vecs_{dnn_str}{PCA_str}.pkl'
    return pickle_wrap(fp_DNN_vecs,
                       lambda: get_DNN_vecs_(PCA=PCA, DNN_layer=DNN_layer),
                       easy_override=False)


def get_img_fns(get_dict=False):
    dir_obj = r'SchemRep_tasks/PTBtasks/updatedObjectsResampled'
    dir_obj = Path(dir_obj)
    dir_scene = r'SchemRep_tasks/PTBtasks/updatedScenesResampled'
    dir_scene = Path(dir_scene)
    fp_stim = r'SchemRep_tasks/PTBtasks/fullStimList.csv'
    fp_stim = Path(fp_stim)
    df_stim = pd.read_csv(fp_stim)

    names = list(df_stim['Object'].values) + list(df_stim['SceneCongruent'].values)
    fps_obj = [f'{dir_obj}/{fn}' for fn in df_stim['ObjectFile'].values]
    fps_scene = [f'{dir_scene}/{fn}' for fn in df_stim['SceneConFilename'].values]
    fps = fps_obj + fps_scene
    img2cat = dict(zip(names, ['obj']*len(fps_obj) + ['scene']*len(fps_scene)))
    if get_dict:
        return dict(zip(names, fps)), img2cat
    else:
        return names, fps
