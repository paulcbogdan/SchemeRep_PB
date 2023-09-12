from pickle_wrap import pickle_wrap
from collections import defaultdict

import numpy as np

from DNN_vectors import get_DNN_vecs, get_img_fns
from organize_bhv import get_trial_info
from nilearn import image, datasets
from glob import glob

from wordvec_get_vectors import get_semantic_vectors
import matplotlib.pyplot as plt
import utils
import scipy.stats as stats

from scipy import io
import pandas as pd
from tqdm import tqdm


def get_stim_RDM(df_sn, d_vecs, obj_only=True, scene_only=False,
                 dif=False):
    # return get_stim_RDM_lifu(df_sn)
    vec_size = len(d_vecs[df_sn['obj'].iloc[0]])
    all_vecs = np.empty((len(df_sn['obj']), vec_size))
    for i, (obj, scene, obj_rename, scene_rename) in enumerate(zip(df_sn['obj'],
            df_sn['scene'], df_sn['obj_rename'], df_sn['scene_rename'])):
        if obj_only:
            # obj = obj.replace(' ', '_')
            vec_obj = d_vecs[obj]
            all_vecs[i, :] = vec_obj
        elif scene_only:
            # scene = scene.replace(' ', '_')
            vec_scene = d_vecs[scene]
            all_vecs[i, :] = vec_scene
        else:
            # obj = obj.replace(' ', '_')
            # scene = scene.replace(' ', '_')
            vec_obj = d_vecs[obj] - d_vecs[scene]
            all_vecs[i, :] = vec_obj
    RDM_stim = np.corrcoef(all_vecs)
    pd.DataFrame(RDM_stim).to_csv('RDM_stim_mine.csv')
    get_stim_RDM_lifu(df_sn)
    # quit()
    return RDM_stim

def get_stim_RDM_lifu(df_sn):
    print('Loading existing...')
    fp_in = r'C:\PycharmProjects_C\SchemeRep\RSAmodels\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
            r'\RSM_VGG16_PCA.mat'
    mat = io.loadmat(fp_in)
    RDM_stim = mat['R']
    RDM_new = np.zeros((len(df_sn), len(df_sn)))

    tblStim = pd.read_csv(r"SchemRep_tasks\PTBtasks\fullStimList.csv")
    tblStim.head()
    filelist = tblStim['ObjectFile'].to_list()
    name2fps = get_img_fns(get_dict=True)

    for obj0 in tqdm(df_sn['obj'], desc='prepping Lifu RDM'):
        obj0 = name2fps[obj0].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
        for obj1 in df_sn['obj']:
            obj1 = name2fps[obj1].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
            idx0 = filelist.index(obj0)
            idx1 = filelist.index(obj1)
            RDM_new[idx0, idx1] = RDM_stim[idx0, idx1]
            RDM_new[idx1, idx0] = RDM_stim[idx1, idx0]
    pd.DataFrame(RDM_new).to_csv('RDM_stim_lifu.csv')
    # quit()

    return RDM_new




def get_atlas_resampled(combine_bilateral=True):
    fp_ref = r'Day2EncSingleTrialModellingLSS_sorted/102/all_ENCruns_sorted/objects/Day2_Run1_Trial4_UnifiedID53_StimID215_Subset2_pairID15_Con3_Resp4_IsObject1.nii'
    img = image.load_img(fp_ref)
    atlas = utils.get_BN_atlas()
    atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                          interpolation='nearest')
    return atlas

def RDM_x_RDM(fMRI_RDM, stim_RDM):
    assert fMRI_RDM.shape == stim_RDM.shape, 'RDMs must be the same shape: ' \
       f'fMRI_RDM.shape = {fMRI_RDM.shape}, stim_RDM.shape = {stim_RDM.shape}'
    tril_idx = np.tril_indices_from(fMRI_RDM, k=-1)
    fMRI_vec = 1 - fMRI_RDM[tril_idx]
    # print(fMRI_vec.shape)
    # quit()
    stim_vec = 1 - stim_RDM[tril_idx]
    # fMRI_M = np.mean(fMRI_vec)
    # return fMRI_M
    r, _ = stats.spearmanr(fMRI_vec, stim_vec)
    # r = np.corrcoef(fMRI_vec, stim_vec)[0, 1]
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

def get_RDM_x_RDM_sn(sn='138', semantic=True):
    PARTS_DO = 100
    if semantic:
        fp_vecs = f'cache/schemerep_sem_vecs_{PARTS_DO}.pkl'
        d_vecs = pickle_wrap(fp_vecs, lambda: get_semantic_vectors(),
                             easy_override=False)
    else:
        fp_vecs = f'cache/schemerep_per_vecs_{PARTS_DO}.pkl'
        d_vecs = pickle_wrap(fp_vecs, lambda: get_DNN_vecs(),
                             easy_override=False)

    df_sn = get_trial_info(sn)
    # print(sorted(df_sn['scene'].values))
    # return .01
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

    # for label, m, t in zip(labels, decoding_M, decoding_t):
    #     print(f'{label}: {m:.3f}, {t:.3f}')
    #
    # plt.errorbar(np.arange(1, N_ROIs + 1), decoding_M, yerr=decoding_SE, fmt='o')
    # plt.show()

def get_all_sns():
    age2sn = defaultdict(list)
    bad_sns = {'126', '131',
               '201', '224', '231', '232', '233', '234', '235'}
    for age in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{age}*_run1.mat'
        fns = glob(bhv_root)
        for fn in fns:
            sn = fn.replace('behavFiles/ENC\\S', '').replace('_run1.mat', '')
            if sn in bad_sns:
                continue
            age2sn[age].append(sn)
    age2sn['healthy'] = age2sn[1] + age2sn[2]
    return age2sn

def do_all(age=1):
    age2sn = get_all_sns()
    print(age2sn[age])
    l = []
    for sn in list(age2sn[age]) + list(age2sn[2]):
        print(f'Doing: {sn}')
        z = get_RDM_x_RDM_sn(sn)
        print(f'{sn}: {z=:.3f}')
        l.append(z)

        if len(l) > 2:
            M = np.mean(l)
            SD = np.std(l)
            SE = SD / np.sqrt(len(l))
            t = M / SE
            print(f'{M=:.3f} [{SE=:.3f}], {t=:.3f}, N = {len(l)}')

def print_ROI_to_results(ROI_to_results):
    print()
    for ROI, results in ROI_to_results.items():
        if len(results) < 2:
            continue
        M = np.mean(results)
        SD = np.std(results)
        SE = SD / np.sqrt(len(results))
        t = M / SE
        print(f'{ROI}: {M=:.3f} [{SE=:.3f}], {t=:.3f}, N = {len(results)}')

def mass_RDM_x_RDM(age='healthy', semantic=False):
    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(early=True, PCA=True)

    # ROIs = ['215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2']
    ROIs = ['189 Cun_L_5_1', '190 Cun_R_5_1', '191 Cun_L_5_2', '192 Cun_R_5_2',
            '193 Cun_L_5_3', '194 Cun_R_5_3', '195 Cun_L_5_4', '196 Cun_R_5_4',
            '197 Cun_L_5_5', '198 Cun_R_5_5']
    ROIs = ['199 OcG_L_4_1', '200 OcG_R_4_1',
            '201 OcG_L_4_2', '202 OcG_R_4_2', '203 OcG_L_4_3', '204 OcG_R_4_3',
            '205 OcG_L_4_4', '206 OcG_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1',
            '209 sOcG_L_2_2', '210 sOcG_R_2_2']
    ROI_nums = [int(roi.split()[0]) for roi in ROIs]
    atlas = get_atlas_resampled()
    age2sn = get_all_sns()

    ROI_to_results = defaultdict(list)
    Zs_all_ = []
    for sn in age2sn[age]:
        df_sn = get_trial_info(sn)
        # df_sn = df_sn[df_sn['CIN'] == 1]
        RDM_stim = get_stim_RDM(df_sn, d_vecs)
        img = image.load_img(df_sn['fp_fMRI']).get_fdata()
        for ROI, ROI_num in zip(ROIs, ROI_nums):
            region_vecs = img[atlas['maps'].get_fdata() == ROI_num]
            RDM_fMRI = np.corrcoef(region_vecs.T)
            z_RDM_x_RDM = RDM_x_RDM(RDM_fMRI, RDM_stim)
            Zs_all_.append(z_RDM_x_RDM)
            if np.isnan(z_RDM_x_RDM):
                print(f'BAD: {sn} {ROI}')
                continue
            ROI_to_results[ROI].append(z_RDM_x_RDM)
        print_ROI_to_results(ROI_to_results)
        Zs_all = np.array(Zs_all_)
        Zs_all = Zs_all[~np.isnan(Zs_all)]
        vmin = -np.max(np.abs(Zs_all))
        vmax = np.max(np.abs(Zs_all))
        plt.hist(Zs_all, range=(vmin, vmax), bins=20)
        plt.show()

if __name__ == '__main__':
    mass_RDM_x_RDM()


