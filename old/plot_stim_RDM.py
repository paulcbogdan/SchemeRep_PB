import random

from sklearn.manifold import MDS
import numpy as np
from PIL import Image, ImageOps
import os

from organize_bhv import get_trial_info, get_all_sns
from stim import get_semantic_vectors, get_DNN_vecs, get_img_fns
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
    pairs_all = {('firetruck', 'music studio'), ('ruler', 'classroom'), ('laundry hamper', 'zoo'),
                 ('haircomb', 'hair salon'), ('horse', 'racetrack'), ('chalice', 'restroom stall'),
                 ('haircomb', 'treehouse'), ('kayak', 'canal'), ('soccerball', 'movie theater'),
                 ('seatbelt', 'library'), ('palm tree', 'shop front'), ('ostrich', 'zoo'),
                 ('laundry hamper', 'laundromat'), ('log', 'treehouse'), ('toilet', 'restroom stall'),
                 ('loudspeaker', 'amphitheater'), ('swimsuit', 'laundromat'), ('slide', 'bus stop'),
                 ('chandelier', 'hotel lobby'), ('flower pot', 'coffee shop'), ('pillow', 'bedroom'),
                 ('dining table', 'castle'), ('steering wheel', 'garage'), ('ambulance', 'pyramid'),
                 ('rollerskates', 'playground'), ('violin', 'concert hall'), ('rollerskates', 'roller rink'),
                 ('hairdryer', 'deli'), ('apple', 'orchard'), ('bench', 'snowy mountains'),
                 ('cutting board', 'post office'), ('potted plant', 'seaport'), ('flowers', 'waves'),
                 ('scorpion', 'airport'), ('reed', 'golf course'), ('soccerball', 'soccer field'),
                 ('spotlight', 'grassland'), ('firetruck', 'bathroom'), ('traffic sign', 'temple'),
                 ('sea gull', 'pier'), ('pine', 'lab'), ('sewing machine', 'sewing room'), ('polar bear', 'iceberg'),
                 ('ostrich', 'amusement park'), ('television', 'hair salon'), ('ice skate', 'country house'),
                 ('popcorn machine', 'movie theater'), ('ambulance', 'hospital'), ('golf club', 'pond'),
                 ('ice skate', 'hotel lobby'), ('cash register', 'ice stadium'), ('sprayer', 'swamp'),
                 ('loudspeaker', 'tropical volcano'), ('car', 'garage'), ('dining chair', 'campsite'),
                 ('sea gull', 'tropical volcano'), ('frame', 'museum'), ('picnic blanket', 'Eiffel Tower'),
                 ('cockroach', 'bar'), ('notebook', 'apartment complex'), ('gavel', 'stage'), ('cash register', 'deli'),
                 ('display cabinet', 'home office'), ('printer', 'post office'), ('sailboat', 'seaport'),
                 ('dog toy', 'restroom stall'), ('rollerskates', 'flower shop'), ('book bag', 'college quad'),
                 ('bench', 'bank'), ('surfing board', 'pier'), ('police car', 'orchard'), ('grill', 'prison'),
                 ('cabin', 'driveway'), ('cheeseburger', 'hair salon'), ('television', 'living room'),
                 ('rock-climbing shoe', 'grocery store'), ('football', 'climbing wall'), ('football', 'football field'),
                 ('seatbelt', 'inside of a car'), ('podium', 'cemetery'), ('police baton', 'greenhouse'),
                 ('truck', 'bridge'), ('ladder', 'attic'), ('grill', 'amphitheater'), ('ostrich', 'gym'),
                 ('scorpion', 'desert'), ('police car', 'police station'), ('mailbox', 'post office'),
                 ('cow', 'grassland'), ('palm tree', 'islands'), ('microscope', 'hospital'), ('seashell', 'bathroom'),
                 ('football', 'bakery'), ('doorknob', 'front porch'), ('oversize tire', 'kitchen'), ('clown', 'arcade'),
                 ('loudspeaker', 'football field'), ('ice skate', 'ice stadium'), ('potted plant', 'balcony'),
                 ('cabin', 'snowy mountains'), ('cross', 'church'), ('police car', 'arch'),
                 ('dragonfly', 'tennis court'), ('microscope', 'lab'), ('flower pot', 'inside of a car'),
                 ('seatbelt', 'bridge'), ('lei', 'tropical volcano'), ('dining chair', 'shop front'),
                 ('office chair', 'courtroom'), ('sewing machine', 'apartment complex'), ('poker table', 'church'),
                 ('ferris wheel', 'orchard'), ('crib', 'living room'), ('horse', 'desert'),
                 ('cheeseburger', "McDonald's"), ('piano', 'buffet'), ('reed', 'pond'), ('goggles', 'swimming pool'),
                 ('beer', 'nursery'), ('cactus', 'waterfall'), ('banana', 'market'), ('briefcase', 'airport'),
                 ('gavel', 'courtroom'), ('beer', 'roller rink'), ('oversize tire', 'monster truck'),
                 ('spotlight', 'lecture hall'), ('cookie', 'arcade'), ('gift bag', 'gas station'),
                 ('flowers', 'kitchen'), ('cactus', 'flower shop'), ('gift bag', 'Eiffel Tower'), ('car', 'canyon'),
                 ('sewing machine', 'playground'), ('dumb bell', 'golf course'), ('cockroach', 'beach'),
                 ('donut', 'conference room'), ('railroad', 'garage'), ('office chair', 'conference room'),
                 ('beer', 'bar'), ('seashell', 'coast'), ('school bus', 'roadside'), ('wheat', 'restaurant'),
                 ('dining chair', 'restaurant'), ('lawn mower', 'apartment complex'), ('cockroach', 'dump'),
                 ('poker table', 'mall'), ('surfing board', 'waves'), ('ambulance', 'pet store'),
                 ('tile', 'football field'), ('gas can', 'lecture hall'), ('wine glass', 'hotel lobby'),
                 ('sprayer', 'greenhouse'), ('wheat', 'barn'), ('dumb bell', 'gym'), ('polar bear', 'pond'),
                 ('cabin', 'bank'), ('tile', 'casino'), ('police baton', 'museum'), ('cheeseburger', 'bus stop'),
                 ('binder clip', 'home office'), ('dining table', 'volleyball court'), ('raft', 'iceberg'),
                 ('toilet', 'woods'), ('traffic sign', 'treehouse'), ('ski', 'concert hall'), ('printer', 'pier'),
                 ('religious statue', 'dump'), ('truck', 'racetrack'), ('railroad', 'train station'), ('ATM', 'deli'),
                 ('surfing board', 'amphitheater'), ('ferris wheel', 'amusement park'), ('camel', 'pyramid'),
                 ('lei', 'bedroom'), ('sea gull', 'front porch'), ('school bus', 'aquarium'), ('violin', 'balcony'),
                 ('kayak', 'waves'), ('ATM', 'bank'), ('ladder', 'fire station'), ('lawn mower', 'country road'),
                 ('police baton', 'prison'), ('game token', 'attic'), ('steering wheel', 'bus'),
                 ('dining table', 'country house'), ('sailboat', 'swimming pool'), ('display cabinet', 'canal'),
                 ('poker table', 'casino'), ('frame', 'climbing wall'), ('rock-climbing shoe', 'park'),
                 ('pillow', 'sauna'), ('wheat', 'cemetery'), ('whistle', 'castle'), ('wine glass', 'pet store'),
                 ('flower pot', 'flower shop'), ('frame', 'office space'), ('apple', 'iceberg'), ('clown', 'hospital'),
                 ('ATM', 'coast'), ('car', 'pyramid'), ('cookie', 'buffet'), ('dragonfly', 'courtroom'),
                 ('podium', 'monster truck'), ('log', 'movie theater'), ('slide', 'zoo'), ('book bag', 'casino'),
                 ('sailboat', 'bus'), ('cutting board', 'bakery'), ('potted plant', 'concert hall'), ('cactus', 'arch'),
                 ('doorknob', 'greenhouse'), ('cookie', 'bus'), ('soccerball', 'laundromat'), ('slide', 'playground'),
                 ('coffee machine', 'coffee shop'), ('television', 'garden'), ('hairdryer', 'grocery store'),
                 ('ski', 'snowy mountains'), ('book bag', 'campsite'), ('piano', 'music studio'), ('toilet', 'lab'),
                 ('game token', 'arcade'), ('swimsuit', 'classroom'), ('cross', 'front porch'),
                 ('tennis ball', 'restaurant'), ('notebook', 'library'), ('school bus', 'balcony'),
                 ('firetruck', 'fire station'), ('jeep', 'waterfall'), ('bench', 'park'), ('polar bear', 'classroom'),
                 ('popcorn machine', 'dump'), ('piano', 'market'), ('briefcase', 'police station'),
                 ('doorknob', 'swamp'), ('ferris wheel', 'beach'), ('scorpion', 'temple'), ('apple', 'gym'),
                 ('helmet', 'home office'), ('jeep', 'soccer field'), ('cash register', 'driveway'),
                 ('tennis ball', 'college quad'), ('mailbox', 'gas station'), ('camel', 'country house'),
                 ('crib', 'nursery'), ('banana', 'roadside'), ('gas can', 'gas station'), ('crib', 'roller rink'),
                 ('gas can', 'barn'), ('pillow', 'Eiffel Tower'), ('dog toy', 'pet store'),
                 ('ruler', 'inside of a car'), ('camel', 'circus'), ('shopping cart', 'circus'),
                 ('briefcase', 'amusement park'), ('railroad', 'woods'), ('raft', 'coffee shop'), ('clown', 'circus'),
                 ('popcorn machine', 'country road'), ('helmet', 'climbing wall'), ('shopping cart', 'grocery store'),
                 ('violin', 'construction site'), ('mailbox', 'bedroom'), ('dog toy', 'ice stadium'),
                 ('palm tree', 'canal'), ('hairdryer', 'bathroom'), ('lei', 'museum'),
                 ('steering wheel', 'swimming pool'), ('golf club', 'police station'), ('dumb bell', 'soccer field'),
                 ('office chair', 'tennis court'), ('laundry hamper', 'sewing room'), ('haircomb', 'nursery'),
                 ('helmet', 'monster truck'), ('lawn mower', 'living room'), ('shopping cart', 'market'),
                 ('rose', 'bridge'), ('religious statue', 'temple'), ('cow', 'mall'), ('rock-climbing shoe', 'canyon'),
                 ('display cabinet', 'shop front'), ('rose', 'garden'), ('seashell', 'fire station'),
                 ('game token', 'buffet'), ('rose', 'library'), ('coffee machine', 'racetrack'),
                 ('chandelier', 'train station'), ('ruler', 'bar'), ('flowers', 'cemetery'),
                 ('construction helmet', 'construction site'), ('construction helmet', 'canyon'),
                 ('cutting board', 'kitchen'), ('chandelier', 'attic'), ('log', 'arch'), ('binder clip', 'barn'),
                 ('reed', 'desert'), ('raft', 'waterfall'), ('chalice', 'winery'), ('oversize tire', 'church'),
                 ('wine glass', 'winery'), ('picnic blanket', 'office space'), ('jeep', 'country road'),
                 ('gavel', 'college quad'), ('tile', 'sauna'), ('seal', 'seaport'), ('podium', 'lecture hall'),
                 ('horse', 'sewing room'), ('whistle', 'construction site'), ('goggles', 'coast'), ('seal', 'winery'),
                 ('microscope', 'train station'), ('banana', 'volleyball court'), ('grill', 'campsite'),
                 ('donut', 'bakery'), ('binder clip', 'prison'), ('religious statue', 'garden'),
                 ('whistle', 'volleyball court'), ('golf club', 'golf course'), ('goggles', 'park'),
                 ('tennis ball', 'tennis court'), ('sprayer', 'stage'), ('truck', "McDonald's"), ('pine', 'driveway'),
                 ('cross', 'sauna'), ('seal', 'aquarium'), ('dragonfly', 'swamp'), ('picnic blanket', 'grassland'),
                 ('ski', 'roadside'), ('notebook', 'airport'), ('pine', 'woods'), ('chalice', 'castle'),
                 ('donut', 'islands'), ('coffee machine', "McDonald's"), ('ladder', 'aquarium'), ('swimsuit', 'beach'),
                 ('cow', 'islands'), ('construction helmet', 'music studio'), ('gift bag', 'mall'),
                 ('printer', 'office space'), ('spotlight', 'stage'), ('traffic sign', 'bus stop'),
                 ('kayak', 'conference room')}
    return pairs_all
    age2sn = get_all_sns(ret=False)
    pairs_all = []
    for sn in age2sn[1]:
        df_sn = get_trial_info(sn)
        pairs = list(zip(df_sn['obj'], df_sn['scene']))
        pairs_all.extend(pairs)
        # break
    pairs_all = set(pairs_all)
    print(f'{pairs_all=}')
    return pairs_all

def get_d_vecs(pairs_all, semantic=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=4, PCA=True, PCA_obj=False)
    d_vec_pairs = {}
    for pair in pairs_all:
        d_vec_pairs[pair] = abs(d_vecs[pair[0]] + d_vecs[pair[1]])
        # d_vec_pairs[pair] = d_vecs[pair[0]] - d_vecs[pair[1]]
    return d_vecs, d_vec_pairs

def filter_zero(data):
    data0 = data[:, :, 0]
    non_white0 = data0 < 255
    data1 = data[:, :, 1]
    non_white1 = data1 < 255
    data2 = data[:, :, 2]
    non_white2 = data2 < 255
    non_white = non_white0 | non_white1 | non_white2
    return data[non_white, :]

def setup_pair_pics(pairs_all, img2fp, avg=True, filt=True):
    pair2fp = {}
    min_val = []
    for pair in pairs_all:
        fn_out = '_'.join(pair)
        avg_str = '_avg' if avg else ''
        fn_out = fn_out.replace(' ', '_') + f'{avg_str}.png'
        fp_out = fr'SchemRep_tasks/combined_pics/{fn_out}'
        pair2fp[pair] = fp_out
        # if os.path.isfile(fp_out):
        #     continue
        img_obj = Image.open(img2fp[pair[0]])
        img_obj = img_obj.resize((600, 600))
        # avg_obj = np.mean
        img_scn = Image.open(img2fp[pair[1]])

        if avg:
            data_obj = np.array(img_obj)
            data_scn = np.array(img_scn)
            if filt:
                vec_obj = filter_zero(data_obj)
                avg_obj = np.mean(vec_obj, axis=0)
                # avg_obj2 = np.mean(data_obj, axis=(0, 1))
                # avg_obj = (avg_obj + avg_obj2)/2
                # avg_grey = np.mean(avg_obj)
                # avg_obj = np.array([avg_grey, avg_grey, avg_grey])
            else:
                avg_obj = np.mean(data_obj, axis=(0, 1))
            # print(vec_obj.shape)
            # data_obj[data_obj == 0] = np.nan
            # vec_scn = filter_zero(data_scn)
            # print(avg_obj.shape)
            # print(avg_obj)
            # quit()
            # avg_scn = np.mean(vec_scn, axis=0)
            avg_scn = np.mean(data_scn, axis=(0, 1))

            # avg_both = (avg_obj - avg_scn) / 2
            # avg_both = 2*(avg_both + 40)
            # print(avg_both)
            # min_val.append(np.min(avg_both))
            # avg_obj = avg_scn = avg_both
            # avg_scn_grey = np.mean(avg_scn)
            # avg_scn = np.array([avg_scn_grey, avg_scn_grey, avg_scn_grey])

            avg_obj = np.repeat(avg_obj[None, None, :], data_obj.shape[0],
                                axis=0)
            avg_obj = np.repeat(avg_obj, data_obj.shape[1], axis=1)
            avg_scn = np.repeat(avg_scn[None, None, :], data_scn.shape[0],
                                axis=0)
            avg_scn = np.repeat(avg_scn, data_scn.shape[1], axis=1)
            img_obj = Image.fromarray(avg_obj.astype('uint8'))
            img_scn = Image.fromarray(avg_scn.astype('uint8'))

        img = Image.new('RGB', (img_obj.width + img_scn.width, img_obj.height))
        img.paste(img_obj, (0, 0))
        img.paste(img_scn, (img_obj.width, 0))
        img = ImageOps.expand(img, border=20, fill='black')
        img.save(fp_out)
    # print(f'{np.mean(min_val)=}')
    # print(f'{np.min(min_val)=}'
    #         f'{np.max(min_val)=}')
    # quit()
    return pair2fp#, pair2cat
    # img2fp.update(pair2fp)
    # for pair in pair2fp:
    #     img2cat[pair] = 'tup'

if __name__ == '__main__':
    AVG_PIC = True
    fp_stim = r'SchemRep_tasks/PTBtasks/fullStimList.csv'
    fp_stim = Path(fp_stim)

    img2fp, img2cat = get_img_fns(get_dict=True)
    pairs_all = get_pairs_all()
    d_vecs, d_vec_pairs = get_d_vecs(pairs_all)
    # d_vecs = d_vec_pairs
    # img2fp = setup_pair_pics(pairs_all, img2fp, avg=AVG_PIC)

    # imgs = []
    imgs = sorted(d_vecs.keys())
    vecs = [d_vecs[img] for img in imgs]
    # for img, vec in d_vecs.items():
    # for img in imgs:
    #     vecs.append(d_vecs[img])
        #if isinstance(img, tuple): continue
        # imgs.append(img)
        # vecs.append(vec)
    vecs = np.array(vecs)
    mds = MDS(n_components=2, random_state=1)
    coords = mds.fit_transform(vecs)
    print(f'{coords.shape=}')
    idxs = list(range(coords.shape[0]))
    fig = plt.figure(dpi=450)
    ax = plt.gca()

    idxs_used = []
    for idx in idxs:
        img = imgs[idx]
        if img2cat[img] != 'obj': continue
        idxs_used.append(idx)
        fp = img2fp[img]
        zoom = .08 if img2cat[img] == 'obj' else .02
        # zoom = .007
        test = plt.imread(fp)

        imagebox = OffsetImage(plt.imread(fp), zoom=zoom, alpha=.75)
        ab = AnnotationBbox(imagebox, coords[idx], frameon=False)
        ax.add_artist(ab)

    plt.axis('off')
    x_min, x_max = min(coords[idxs_used, 0]), max(coords[idxs_used, 0])
    y_min, y_max = min(coords[idxs_used, 1]), max(coords[idxs_used, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    print(f'{x_min=:.3f}, {y_min=:.3f}')
    plt.show()





