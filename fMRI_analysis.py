from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np
from prep_names import organize_subj_df
from nilearn import image, datasets
from glob import glob

from wordvec_get_vectors import get_vectors
import matplotlib.pyplot as plt

def get_stim_RDM(df_sn, d_vecs):
    # print(df_sn['obj'].iloc[0])
    vec_size = len(d_vecs[df_sn['obj'].iloc[0].replace(' ', '_')])
    all_vecs = np.empty((len(df_sn['obj']), vec_size))
    for i, (obj, scene) in enumerate(zip(df_sn['obj'],
                                         df_sn['scene'])):
        obj = obj.replace(' ', '_')
        scene = scene.replace(' ', '_')
        # vec_obj = d_vecs[obj]# - d_vecs[scene]
        vec_obj = d_vecs[scene]

        all_vecs[i, :] = vec_obj
    RDM_stim = np.corrcoef(all_vecs)
    return RDM_stim

def get_atlas_resampled(img, combine_bilateral=True):
    # dataset_ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm",
    #                                                  symmetric_split=False)
    dataset_ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm",
                                                     symmetric_split=True)


    # print('Labels:')
    # for i, label in enumerate(dataset_ho.labels):
    #     print(f'{i} = {label}')
    # quit()
    # print(dataset_ho.maps == 3)
    # print(dataset_ho.maps.dataobj.shape)

    # if combine_bilateral:
    data = dataset_ho.maps.dataobj
    data[data == 69] = 255
    data[data == 70] = 255


    # quit()

    # if combine_bilateral:
    #     dataset_ho.maps

    dataset_ho.maps = image.resample_to_img(dataset_ho.maps,
                                            img, interpolation='nearest')
    return dataset_ho

def RDM_x_RDM(fMRI_RDM, stim_RDM):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    fMRI_vec = fMRI_RDM[tril_idx]
    stim_vec = stim_RDM[tril_idx]
    r = np.corrcoef(fMRI_vec, stim_vec)[0, 1]
    z = np.arctanh(r)
    return z

def explore_hist(stim_RDM):
    tril_idx = np.tril_indices_from(stim_RDM, k=-1)
    stim_vec = stim_RDM[tril_idx]
    vmin = np.quantile(stim_vec, .005)
    vmax = np.quantile(stim_vec, .995)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].imshow(stim_RDM, vmin=vmin, vmax=vmax, cmap='turbo')
    axs[1].hist(stim_vec, bins=100, range=(vmin, vmax))
    plt.show()

def get_RDM_x_RDM_sn(sn='138'):
    fp_vecs = 'cache/schemerep_sim_vecs.pkl'
    d_vecs = pickle_wrap(fp_vecs, lambda: get_vectors(),
                         easy_override=False)
    df_sn = organize_subj_df(sn)
    RDM_stim = get_stim_RDM(df_sn, d_vecs)
    cnt_RDM_stim_nan = np.sum(np.isnan(RDM_stim))
    print(f'{cnt_RDM_stim_nan=}')
    img = image.load_img(df_sn['fp_fMRI'])
    atlas = get_atlas_resampled(img)
    img_data = img.get_fdata()
    ROI_num = 255
    region_vecs = img_data[atlas.maps.get_fdata() == ROI_num]
    bad_voxels = np.unique(np.argwhere(np.isnan(region_vecs))[:, 0])
    if len(bad_voxels):
        print(f'{sn} has {len(bad_voxels)} bad voxels')
    region_vecs =  np.delete(region_vecs, bad_voxels, axis=0)

    RDM_fMRI = np.corrcoef(region_vecs.T)
    # cnt_fMRI_nan = np.sum(np.isnan(region_vecs))

    z_RDM_x_RDM = RDM_x_RDM(RDM_fMRI, RDM_stim)
    if np.isnan(z_RDM_x_RDM):
        plt.imshow(np.isnan(region_vecs))
        plt.show()
        print(f'NaN: {sn=}')
        quit()
    return z_RDM_x_RDM

def get_all_sns():
    age2sn = defaultdict(list)
    for i in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{i}*_run1.mat'
        fns = glob(bhv_root)
        for fn in fns:
            sn = fn.replace('behavFiles/ENC/S', '').replace('_run1.mat', '')
            age2sn[i].append(sn)
    return age2sn

def do_all(age=1):
    age2sn = get_all_sns()
    print(age2sn[age])
    l = []
    for sn in list(age2sn[age]) + list(age2sn[2]):
        print(f'Doing: {sn}')
        if sn == '126': continue # missing
        if sn == '131': continue

        if sn == '201': continue
        if sn == '224': continue
        if sn == '231': continue
        if sn == '232': continue
        if sn == '233': continue
        if sn == '234': continue
        if sn == '235': continue
        # if sn != '129': continue
        z = get_RDM_x_RDM_sn(sn)
        print(f'{sn}: {z=:.3f}')
        l.append(z)

        if len(l) > 2:
            M = np.mean(l)
            SD = np.std(l)
            SE = SD / np.sqrt(len(l))
            t = M / SE
            print(f'{M=:.3f} [{SE=:.3f}], {t=:.3f}, N = {len(l)}')

if __name__ == '__main__':
    do_all()






