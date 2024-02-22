from pathlib import Path

# import gensim
import numpy as np
import pandas as pd
from PIL import Image
from utils import pickle_wrap
from sklearn import decomposition
# try:
#     from torchvision import models as models, transforms as transforms
#     import torch
# except OSError:
#     # Can't get this to work on my py3.11
#     pass

import scipy.stats as stats
#from old.test_lifu import get_stim_RDM_lifu
from organize_bhv import get_trial_info
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: check, U:\Cabeza\SchemRep.01\Scripts\RSA\RSAmodels\RSM_VGG16_PCA.mat
# Lifu used it, per analysis_v2_ENC_bars.m

def get_stim_RDM_lifu(df_sn, per=True):
    from tqdm import tqdm
    import scipy.io as io
    print('Loading existing...')
    if per:
        fp_in = r'E:\PycharmProjects_E\SchemeRep\old\RSAmodels' \
                r'\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
                r'\RSM_VGG16_PCA.mat'
    else:
        fp_in = r'E:\PycharmProjects_E\SchemeRep\old\RSAmodels\W2Vsemantic_RDM.mat'
    mat = io.loadmat(fp_in)
    RDM_stim = mat['R']
    RDM_new = np.zeros((len(df_sn), len(df_sn)))

    tblStim = pd.read_csv(r"SchemRep_tasks\PTBtasks\fullStimList.csv")
    tblStim.head()
    filelist = tblStim['ObjectFile'].to_list()
    name2fps, _ = get_img_fns(get_dict=True)

    for obj0 in tqdm(df_sn['obj'], desc='prepping Lifu RDM'):
        obj0 = name2fps[obj0].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
        for obj1 in df_sn['obj']:
            obj1 = name2fps[obj1].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
            idx0 = filelist.index(obj0)
            idx1 = filelist.index(obj1)
            RDM_new[idx0, idx1] = RDM_stim[idx0, idx1]
            RDM_new[idx1, idx0] = RDM_stim[idx1, idx0]
    # pd.DataFrame(RDM_new).to_csv('RDM_stim_lifu.csv')

    return RDM_new


def get_stim_RDM(df_sn, d_vecs, obj_only=False, scene_only=False,
                 dif=False, prod=False, take_abs=False, add=False,
                 lifu=False, lifu_sem=False, dist='corr'):
    assert obj_only or scene_only or dif or prod or add or lifu or lifu_sem, \
        'Must specify one of obj_only, scene_only, dif, prod, add, lifu'
    if lifu or lifu_sem:
        return get_stim_RDM_lifu(df_sn, per=not lifu_sem)
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
    # print(f'{all_vecs.shape=}')
    # quit()
    if dist == 'corr':
        RDM_stim = np.corrcoef(all_vecs)
    elif dist == 'spear':
        RDM_stim = np.zeros((all_vecs.shape[0], all_vecs.shape[0]))
        for i in range(all_vecs.shape[0]):
            for j in range(all_vecs.shape[0]):
                if i >= j:
                    continue
                r, p = stats.spearmanr(all_vecs[i], all_vecs[j])
                RDM_stim[i, j] = r
                RDM_stim[j, i] = r
        RDM_stim[np.diag_indices_from(RDM_stim)] = np.nan
    elif dist == 'mahalanobis' or dist == 'seuclidean':
        RDM_stim = scipy_dist(all_vecs, metric=dist)
    elif dist == 'euc':
        RDM_stim = scipy_dist(all_vecs, metric='euclidean')
    else:
        raise ValueError('dist must be corr or mahalanobis')
    return RDM_stim

def scipy_dist(all_vecs, vecs1=None, metric='seuclidean'):
    from scipy.spatial import distance
    all_vecs = np.array(all_vecs)
    # V_by_edge = np.nanvar(all_vecs, axis=0)
    # RSM = np.zeros((all_vecs.shape[0], all_vecs.shape[0]))
    # for i in range(all_vecs.shape[0]):
    #     for j in range(all_vecs.shape[1]):
    #         vec_i = all_vecs[i]
    #         vec_j = all_vecs[j]
    #         r = distance.seuclidean(vec_i, vec_j, V_by_edge)
    #         RSM[i, j] = r
    # RSM[np.diag_indices(all_vecs.shape[0])] = np.nan
    # plt.imshow(RSM)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(all_vecs)
    # plt.show()
    # print(np.sum(np.isnan(all_vecs)))
    # print(np.sum(np.isnan(vecs1)))
    # quit()
    if vecs1 is None:
        RSM_triangle = -distance.pdist(all_vecs, metric)
        RSM = distance.squareform(RSM_triangle)
        RSM[np.diag_indices(all_vecs.shape[0])] = np.nan
    else:
        vecs1 = np.array(vecs1)
        # print(f'{all_vecs.shape=}')
        # print(f'{vecs1.shape=}')
        RSM = -distance.cdist(all_vecs, vecs1, metric)

    #     plt.imshow(RSM)
    #     plt.show()
    #     quit()
    # # print(RSM.shape)
    # RSM = prune_RSM_outliers(RSM)
    # plt.imshow(RSM)
    # plt.colorbar()
    # plt.show()
    # quit()
    # impute mean for nan

    return RSM

def prune_RSM_outliers(RSM, z=3):
    prune_outliers = True
    while prune_outliers:
        M_dis = np.nanmean(RSM, axis=0)
        Z_M_dis = (M_dis - np.nanmean(M_dis)) / np.nanstd(M_dis)
        M_dis1 = np.nanmean(RSM, axis=1)
        Z_M_dis1 = (M_dis1 - np.nanmean(M_dis1)) / np.nanstd(M_dis1)

        if np.nanmax(np.abs(Z_M_dis)) > z:
            # print('Pruning outliers')
            outliers = np.argwhere(np.abs(Z_M_dis) > z)
            # print(f'{outliers=}')
            RSM[outliers, :] = np.nan
            RSM[:, outliers] = np.nan
        elif np.nanmax(np.abs(Z_M_dis1)) > z:
            # print('Pruning outliers')
            outliers = np.argwhere(np.abs(Z_M_dis1) > z)
            # print(f'{outliers=}')
            RSM[outliers, :] = np.nan
            RSM[:, outliers] = np.nan
        else:
            prune_outliers = False
    return RSM

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


def get_semantic_vectors_(normalize=False, norm_by_type=True):
    from gensim import downloader
    w2vectors = downloader.load('word2vec-google-news-300')
    print('Loaded word2vec')
    df = get_trial_info('138')
    d_all = {}
    vecs_obj = []
    vecs_scn = []
    for obj, scene, obj_rename, scene_rename in tqdm(zip(df['obj'], df['scene'],
                          df['obj_rename'], df['scene_rename']),
                          desc='Getting semantic vectors'):
        d_all[obj] = get_vec(obj_rename, w2vectors)
        d_all[scene] = get_vec(scene_rename, w2vectors)
        vecs_obj.append(d_all[obj])
        vecs_scn.append(d_all[scene])
        # vecs_all.append(d_all[obj])
        # vecs_all.append(d_all[scene])
        # print(f'{d_all[obj]=}')

    if normalize:
        d_all = norm_vectors(d_all, vecs_obj, vecs_scn, norm_by_type,
                             set(df['obj'].values))
    return d_all

def norm_vectors(d_all, vecs_obj, vecs_scn, norm_by_type, objs):
    if norm_by_type:
        vecs_SDs_obj = np.nanstd(vecs_obj, axis=0)
        vecs_Ms_obj = np.nanmean(vecs_obj, axis=0)
        vecs_SDs_scn = np.nanstd(vecs_scn, axis=0)
        vecs_Ms_scn = np.nanmean(vecs_scn, axis=0)
        for name, vec in d_all.items():
            if name in objs:
                vec -= vecs_Ms_obj
                vec /= vecs_SDs_obj
            else:
                vec -= vecs_Ms_scn
                vec /= vecs_SDs_scn
            d_all[name] = vec
    else:
        vecs_all = np.array(vecs_obj + vecs_scn)
        print(np.sum(np.isnan(vecs_all)))
        plt.imshow(vecs_all)
        plt.show()
        vec_SDs = np.nanstd(vecs_all, axis=0)
        vec_Ms = np.nanmean(vecs_all, axis=0)
        for name, vec in d_all.items():
            vec -= vec_Ms
            vec /= vec_SDs
            d_all[name] = vec
    return d_all

def get_semantic_vectors(normalize=True):
    # fit using python 3.11
    norm_string = '_norm' if normalize else ''
    fp_vecs = f'cache/schemerep_sem_vecs{norm_string}.pkl'
    d_vecs = pickle_wrap(lambda: get_semantic_vectors_(normalize), fp_vecs, easy_override=False)
    return d_vecs


def get_DNN_vecs_(PCA=False, DNN_layer=2, PCA_obj=False):
    model = models.vgg16(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(model.features)

    data_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])
                    ])

    names, fps = get_img_fns()
    img_vecs = []
    scn_vecs = []
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
            # from copy import deepcopy
            # x_ = deepcopy(x)
            # for j in range(len(model.features)):
            #     x_ = model.features[j](x_)
            #     print(f'{j} | {x.shape=}')
            # x_ = model.avgpool(x_)
            # x_ = torch.flatten(x_, 1)
            # x_ = model.classifier(x_)
            # print(f'{x_}')
            # print(f'{x_.shape=}')
            x = model.forward(x)
            # print(f'Test: {x.shape=}')
        else:
            for j in range(DNN_layer):
                x = model.features[j](x)
        # if early:
        #     x = model.features[0](x)
        #     x = model.features[1](x)
        # else:
        #     x = model.forward(x)
        x = x.detach().numpy().flatten()
        # x = np.concatenate([x, np.array([scene])]) # Why do I have this last feat?
        img_vecs.append(x)
        if not scene:
            obj_vecs.append(x)
        else:
            scn_vecs.append(x)
        names_sanity.append(name)

    scn_vecs = np.array(scn_vecs)
    img_vecs = np.array(img_vecs)
    obj_vecs = np.array(obj_vecs)

    if PCA:
        pca = decomposition.PCA()
        if PCA_obj:
            pca.fit(obj_vecs)
        else:
            # pca.fit(scn_vecs)
            pca.fit(img_vecs)
        img_vec_brief = pca.transform(img_vecs)
        img_vec_brief = img_vec_brief
    else:
        img_vec_brief = img_vecs

    d_vecs = {}
    for name, vec in zip(names_sanity, img_vec_brief):
        d_vecs[name] = vec
    return d_vecs


def get_DNN_vecs(PCA=True, PCA_obj=True, DNN_layer=2, easy_override=False):
    dnn_str = 'late' if DNN_layer == -1 else \
              'early' if DNN_layer == 2 else \
              f'dnn{DNN_layer}'
    PCA_str = '_PCA' if PCA else ''
    PCA_obj_str = '' if PCA_obj else '_PCAallImg'
    fp_DNN_vecs = fr'cache/DNN_vecs_{dnn_str}{PCA_str}{PCA_obj_str}.pkl'
    return pickle_wrap(lambda: get_DNN_vecs_(PCA=PCA, DNN_layer=DNN_layer,
                                             PCA_obj=PCA_obj), fp_DNN_vecs, easy_override=easy_override)


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

if __name__ == '__main__':
    d_vecs = get_DNN_vecs(PCA=True, PCA_obj=True, DNN_layer=5,
                          easy_override=True)
    for obj, vec in d_vecs.items():
        print(f'{obj} | {vec.shape=}')
    # quit()
    #
    # d_vecs = get_semantic_vectors(normalize=True)
    # df = get_trial_info('138')
    # vecs_all = []
    # for obj in df['obj']:
    #     vec = d_vecs[obj]
    #     vecs_all.append(vec)
    # # for key, vec in d_vecs.items():
    # #     vecs_all.append(vec)
    # vecs_all = np.array(vecs_all)
    # print(vecs_all.shape)
    # M_vec = np.mean(vecs_all, axis=0)
    # print(f'{M_vec.shape=}')
    # SD_vec = np.std(vecs_all, axis=0)
    # print(SD_vec)
    # plt.hist(M_vec)
    # plt.title('Mean vec')
    # plt.show()
    # plt.hist(SD_vec)
    # plt.title('SD vec')
    # plt.show()
    # quit()


    # all_vecs =  [(35.0456, -85.2672),
    #       (35.1174, np.nan),
    #       (np.nan, -83.9422),
    # #       (36.1667, -86.7833)]
    # all_vecs = [(35.0456, -85.2672),
    #       (35.1174, -89.9711),
    #       (35.9728, -83.9422),
    #       (36.1667, -86.7833)]
    #
    # # all_vecs = [(35.0456, -85.2672),
    # #       (35.1174, -89.9711),
    # #       (35.9728, -83.9422),
    # #       (36.1667, -86.7833),
    # #      (35.0456, -85.2672),
    # #      (35.1174, -89.9711),
    # #      (35.9728, -83.9422),
    # #      (36.1667, -86.7833)
    # #      ]
    # scipy_dist(all_vecs, 'seuclidean')
    # # d_vecs = get_DNN_vecs()
    # d_vecs = get_semantic_vectors()