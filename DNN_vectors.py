from pickle_wrap import pickle_wrap

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
from sklearn import decomposition

# The first hidden layer generated a sensory model, since this model is derived from a layer that detects sensory features, and the penultimate layer, a categorical model, since this model is derived from the layer before the images are explicitly categorized into the trained categories


def get_DNN_vecs_(early=True, PCA=False):
    if early:
        # vec_size = 64*55*55
        vec_size = 64*224*224
        # model = models.alexnet(pretrained=True)
        model = models.vgg16(pretrained=True)
    else:
        vec_size = 1000
        model = models.vgg16(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # data_transforms = transforms.Compose([
    #     transforms.Resize((224,224)),             # resize the input to 224x224
    #     transforms.ToTensor(),              # put the input to tensor format
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
    #     the normalization is based on images from ImageNet
    # ])

    data_transforms = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])

    names, fps = get_img_fns()
    names, fps = zip(*[(name, fp) for name, fp in zip(names, fps)
                       if 'Scene' not in fp])

    # img_vecs = np.empty((vec_size, len(fps)))
    img_vecs = []
    names_sanity = []

    layer1dat = np.zeros((64 * 224 * 224, len(fps)))

    for i, (name, fp) in enumerate(zip(names, fps)):
        # if 'Scene' in fp:
        #     continue
        img = Image.open(fp)
        # print(f'{name} | {fp}')
        img = np.array(img, dtype=np.uint8)
        x = data_transforms(img).unsqueeze(0)
        if early:
            x = model.features[0](x)
            x = model.features[1](x)
        else:
            x = model.forward(x)
        img_vecs.append(x.detach().numpy().flatten())
        names_sanity.append(name)
        # layer1dat[:, i] = x.detach().numpy().flatten()
    img_vecs = np.array(img_vecs)
    # img_vecs = img_vecs.T

    if PCA:
        # print('1:', layer1dat.shape)
        pca = decomposition.PCA()
        # pca.fit(layer1dat.T)
        img_vec_brief = pca.fit_transform(img_vecs)
        img_vec_brief = img_vec_brief
        # print(f'{layer1dat=}')
        # img_vec_brief = pca.transform(layer1dat.T)
        # img_vec_brief = pca.transform(img_vecs.T)
        print(f'{img_vec_brief=}')
        print(f'{img_vec_brief.shape=}')
        # print(pca.components_)
    else:
        img_vec_brief = img_vecs

    # RSM_skl = np.corrcoef(img_vec_brief)
    # print(f'{RSM_skl.shape=}')
    # plt.imshow(RSM_skl)
    # plt.title(f'{img_vec_brief=}')
    # plt.show()
    # print(f'{RSM_skl=}')
    # mean = np.mean(np.tril(RSM_skl, k=-1))
    # print(f'{mean=}')
    # pd.DataFrame(RSM_skl).to_csv('RSM_skl.csv')
    # quit()

    # print('components:', pca.components_)
    # quit()
    # print(img_vec_brief.shape)
    # quit()

    d_vecs = {}
    for name, vec in zip(names_sanity, img_vec_brief):
        d_vecs[name] = vec
    print('test rock:', d_vecs['rock-climbing shoe'])
    print(f'{len(d_vecs)=}')
    # quit()

    return d_vecs

def get_DNN_vecs(early=True, PCA=False):
    early_late_str = 'early' if early else 'late'
    PCA_str = '_PCA' if PCA else ''
    fp_DNN_vecs = fr'cache/DNN_vecs_{early_late_str}{PCA_str}.pkl'
    return pickle_wrap(fp_DNN_vecs, lambda: get_DNN_vecs_(early=early,
                                                          PCA=PCA),
                       easy_override=True)

def get_img_fns(get_dict=False):
    dir_obj = r'SchemRep_tasks\PTBtasks\updatedObjectsResampled'
    dir_scene = r'SchemRep_tasks\PTBtasks\updatedScenesResampled'
    df_stim = pd.read_csv(r'SchemRep_tasks\PTBtasks/fullStimList.csv')

    names = list(df_stim['Object'].values) + list(df_stim['SceneCongruent'].values)
    fps_obj = [f'{dir_obj}/{fn}' for fn in df_stim['ObjectFile'].values]
    fps_scene = [f'{dir_scene}/{fn}' for fn in df_stim['SceneConFilename'].values]
    fps = fps_obj + fps_scene
    if get_dict:
        return dict(zip(names, fps))
    else:
        return names, fps

if __name__ == '__main__':
    get_DNN_vecs(early=True, PCA=True)
