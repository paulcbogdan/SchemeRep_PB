import random

from sklearn.manifold import MDS
import numpy as np
from PIL import Image, ImageOps
import os

from DNN_vectors import get_DNN_vecs, get_img_fns
from organize_bhv import get_trial_info, get_all_sns
from wordvec_get_vectors import get_semantic_vectors
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib as mpl

# TODO: for the objects, maybe adjust how zoomed in some of them are,
#   so that all roughly cover the same amount of whitespace?

def do_dif(d_vecs, pairs_all):
    for pair in pairs_all:
        obj, scn = pair
        fp_obj = img2fp[obj]
        fp_scn = img2fp[scn]
        img_obj = plt.imread(fp_obj)
        img_scn = plt.imread(fp_scn)
        img = np.hstack([img_obj, img_scn])
        img[0, :, :] = img[-1, :, :] = 0
        img[:, 0, :] = img[:, -1, :] = 0
        img[:, :, 0] = img[:, 0, -1] = 0
        imagebox = OffsetImage(plt.imread(fp), zoom=zoom, alpha=.7)
        ab = AnnotationBbox(imagebox, coords[idx], frameon=False)
        ax.add_artist(ab)

def get_pairs_all():
    age2sn = get_all_sns(ret=False)
    pairs_all = []
    for sn in age2sn[1]:
        df_sn = get_trial_info(sn)
        pairs = list(zip(df_sn['obj'], df_sn['scene']))
        pairs_all.extend(pairs)
        break
    pairs_all = set(pairs_all)
    return pairs_all

def get_d_vecs(pairs_all, include_pairs=True, semantic=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True)
    d_vec_pairs = {}
    for pair in pairs_all:
        d_vec_pairs[pair] = abs(d_vecs[pair[0]] - d_vecs[pair[1]])
    return d_vecs, d_vec_pairs

def setup_pair_pics(pairs_all, img2fp, img2cat):
    pair2fp = {}
    for pair in pairs_all:
        fn_out = '_'.join(pair)
        fn_out = fn_out.replace(' ', '_') + '.png'
        fp_out = fr'SchemRep_tasks/combined_pics/{fn_out}'
        pair2fp[pair] = fp_out
        if os.path.isfile(fp_out):
            continue
        img_obj = Image.open(img2fp[pair[0]])
        img_obj = img_obj.resize((600, 600))
        img_scn = Image.open(img2fp[pair[1]])

        img = Image.new('RGB', (img_obj.width + img_scn.width, img_obj.height))
        img.paste(img_obj, (0, 0))
        img.paste(img_scn, (img_obj.width, 0))
        img = ImageOps.expand(img, border=20, fill='black')
        img.save(fp_out)
    return pair2fp#, pair2cat
    # img2fp.update(pair2fp)
    # for pair in pair2fp:
    #     img2cat[pair] = 'tup'

if __name__ == '__main__':
    fp_stim = r'SchemRep_tasks/PTBtasks/fullStimList.csv'
    fp_stim = Path(fp_stim)
    print(fp_stim)
    df_stim = pd.read_csv(fp_stim)

    img2fp, img2cat = get_img_fns(get_dict=True)
    pairs_all = get_pairs_all()
    d_vecs, d_vec_pairs = get_d_vecs(pairs_all)
    d_vecs = d_vec_pairs
    img2fp = setup_pair_pics(pairs_all, img2fp, img2cat)

    imgs = []
    vecs = []
    for img, vec in d_vecs.items():
        #if isinstance(img, tuple): continue
        imgs.append(img)
        vecs.append(vec)
    vecs = np.array(vecs)
    mds = MDS(n_components=2, normalized_stress='auto')
    coords = mds.fit_transform(vecs)
    idxs = list(range(coords.shape[0]))
    fig = plt.figure(dpi=300)
    ax = plt.gca()

    idxs_used = []
    for idx in idxs:
        img = imgs[idx]
        #if img2cat[img] == 'scene': continue
        idxs_used.append(idx)
        fp = img2fp[img]
        #zoom = .08 if img2cat[img] == 'obj' else .03
        zoom = .015
        test = plt.imread(fp)

        imagebox = OffsetImage(plt.imread(fp), zoom=zoom, alpha=.7)
        ab = AnnotationBbox(imagebox, coords[idx], frameon=False)
        ax.add_artist(ab)

    plt.axis('off')
    x_min, x_max = min(coords[idxs_used, 0]), max(coords[idxs_used, 0])
    y_min, y_max = min(coords[idxs_used, 1]), max(coords[idxs_used, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    print(f'{x_min=:.3f}, {y_min=:.3f}')
    plt.show()





