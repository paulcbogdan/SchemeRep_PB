import numpy as np
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.regions import RegionExtractor
from nilearn.regions import img_to_signals_labels
from nilearn.image import resample_to_img, load_img

# Need subject

dataset_ho = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
# print(dataset_ho.labels)
# for i, label in enumerate(dataset_ho.labels):
#     print(f'{i}: {label}')

# 0: Background
# 1: Left Cerebral White Matter
# 2: Left Cerebral Cortex
# 3: Left Lateral Ventrical
# 4: Left Thalamus
# 5: Left Caudate
# 6: Left Putamen
# 7: Left Pallidum
# 8: Brain-Stem
# 9: Left Hippocampus
# 10: Left Amygdala
# 11: Left Accumbens
# 12: Right Cerebral White Matter
# 13: Right Cerebral Cortex
# 14: Right Lateral Ventricle
# 15: Right Thalamus
# 16: Right Caudate
# 17: Right Putamen
# 18: Right Pallidum
# 19: Right Hippocampus
# 20: Right Amygdala
# 21: Right Accumbens

def extract_ROI(img, labels_img, ROI_num):
    img_data = img.get_fdata()
    extracted = img_data[labels_img.get_fdata() == ROI_num]
    return extracted

def compute_RDM(img, labels_img, ROI_num):
    extracted = extract_ROI(img, labels_img, ROI_num)
    RDM = np.corrcoef(extracted.T)
    return RDM


fp_in = r'G:\preprocessing\HCP\HCP_WM_data\100206\MNINonLinear\Results\tfMRI_WM_LR\tfMRI_WM_LR.nii'
img = load_img(fp_in)
# print(dataset_ho.maps.shape)

dataset_ho.maps = resample_to_img(dataset_ho.maps, img, interpolation='nearest')
# extract_ROI(img, dataset_ho.maps, 9)
rdm = compute_RDM(img, dataset_ho.maps, 9)
# print(dataset_ho.maps.shape)
# test, labels = img_to_signals_labels(fp_in, dataset_ho.maps)
# print(test.shape)
# print(labels)

def get_fp_in(sn):
    pass

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
    axs[0].imshow(stim_RDM, vmin=vmin, vmax=vmax, color='turbo')
    axs[1].hist(stim_vec, bins=100, range=(vmin, vmax))
    plt.show()

def analyze_stim_RDM(stim_RDM):
    sns = []
    labels = dataset_ho.labels
    N_ROIs = len(labels)
    decoding_scores = np.empty((N_ROIs, len(sns)))
    dataset_ho.maps = resample_to_img(dataset_ho.maps, get_fp_in(sns[0]),
                                      interpolation='nearest')
    for sn in sns:
        fp_in = get_fp_in(sn)
        img = load_img(fp_in)
        for i in range(1, N_ROIs + 1):
            fMRI_RDM = compute_RDM(img, dataset_ho.maps, 9)
            decoding_score = RDM_x_RDM(fMRI_RDM, stim_RDM)
            decoding_scores[i - 1, sn] = decoding_score

    decoding_M = np.mean(decoding_scores, axis=1)
    decoding_SE = np.std(decoding_scores, axis=1) / np.sqrt(len(sns))
    decoding_t = decoding_M / decoding_SE
    for label, m, t in zip(labels, decoding_M, decoding_t):
        print(f'{label}: {m:.3f}, {t:.3f}')

    plt.errorbar(np.arange(1, N_ROIs + 1), decoding_M, yerr=decoding_SE, fmt='o')
    plt.show()
