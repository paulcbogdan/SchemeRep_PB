from organize_bhv import get_trial_info, NAME_RENAMER
import pandas as pd
from PIL import Image

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
from sklearn.decomposition import PCA

# The first hidden layer generated a sensory model, since this model is derived from a layer that detects sensory features, and the penultimate layer, a categorical model, since this model is derived from the layer before the images are explicitly categorized into the trained categories

def get_VGG_text(x, model):
    x = batch_img
    # print(len(model.features))
    for i in range(len(model.features)):
        x = model.features[i](x)
        print(f'{i}: {type(model.features[i])} | {x.shape}')
    print('-' * 50)
    x = model.avgpool(x)
    print(f'avg pool: {type(model.avgpool)} | {x.shape}')
    x = torch.flatten(x, 1)
    print(f'flatten: {type(model.avgpool)} | {x.shape}')
    for i in range(len(model.classifier)):
        x = model.classifier[i](x)
        print(f'{i}: {type(model.classifier[i])} | {x.shape}')
    g = model.forward(batch_img)
    print(f'meh: {g.shape}')

def get_DNN_vecs(VGG_final=False, alexnet_first=True):
    if alexnet_first:
        vec_size = 64*55*55
        model = models.alexnet(pretrained=True)
    else:
        vec_size = 1000
        model = models.vgg16(pretrained=True)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    names, fps = get_names_fps()

    img_vecs = np.empty((vec_size, len(fps)))

    for fp in fps:
        img = Image.open(fp)
        transformed_img = data_transforms(img)
        x = torch.unsqueeze(transformed_img, 0)
        if alexnet_first:
            x = model.features[0](x)
            x = model.features[1](x)
        else:
            x = model.forward(x)
        img_vecs[:, fps.index(fp)] = x.detach().numpy().flatten()

    pca = PCA(n_components=10)
    pca.fit(img_vecs.T)
    img_vec_brief = pca.transform(img_vecs.T)

    d_vecs = {}
    for name, vec in zip(names, img_vec_brief):
        d_vecs[name] = vec

    return d_vecs

def get_names_fps():
    df = get_trial_info('138')
    names, fps = [], []
    dir_obj = r'SchemRep_tasks\PTBtasks\updatedObjectsResampled'
    dir_scene = r'SchemRep_tasks\PTBtasks\updatedScenesResampled'

    df_stim = pd.read_csv(r'SchemRep_tasks\PTBtasks/fullStimList.csv')
    # df_obj = pd.DataFrame({'fn': df['ObjectFile'].values},
    #                           index=df['Object'].values)
    # scene_l = list(df['SceneCongruent'].values)# + \
    #           # list(df['SceneIncongruent'].values) + \
    #           # list(df['SceneNeutral'].values)
    # scene_fn_l = list(df['SceneConFilename'].values)# + \
    #              # list(df['SceneIncFilename'].values) + \
    #              # list(df['SceneNeuFilename'].values)
    # df_scene = pd.DataFrame({'fn': scene_fn_l}, index=scene_l)
    # df_fns = pd.concat([df_obj, df_scene])

    # names = list(df_stim['Object'].values) + \
    #         list(df_stim['SceneCongruent'].values)
    # fns = list(df_stim['ObjectFile'].values) + \
    #       list(df_stim['SceneConFilename'].values)
    # d_name2fn = dict(zip(names, fns))
    d_obj_name2fn = dict(zip(df_stim['Object'].values,
                             df_stim['ObjectFile'].values))
    d_scene_name2fn = dict(zip(df_stim['SceneCongruent'].values,
                               df_stim['SceneConFilename'].values))

    # renamer_flipped = dict((rename, name) for name, rename in NAME_RENAMER.items())
    # df_scene_idx = pd.DataFrame({'scene': df[]})

    # print(d_name2fn)

    for obj, scene in zip(df['obj'], df['scene']):
        # obj_OG_name = renamer_flipped.get(obj, obj)
        names.append(obj)
        fps.append(f'{dir_obj}/{d_obj_name2fn[obj]}')

        # scene_OG_name = renamer_flipped.get(scene, scene)
        names.append(scene)
        fps.append(f'{dir_scene}/{d_scene_name2fn[scene]}')
    return names, fps

if __name__ == '__main__':
    get_DNN_vecs()