from pickle_wrap import pickle_wrap

from organize_bhv import get_trial_info, NAME_RENAMER
import pandas as pd
from PIL import Image
from pathlib import Path

import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
from sklearn import decomposition

# The first hidden layer generated a sensory model, since this model is derived from a layer that detects sensory features, and the penultimate layer, a categorical model, since this model is derived from the layer before the images are explicitly categorized into the trained categories


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
