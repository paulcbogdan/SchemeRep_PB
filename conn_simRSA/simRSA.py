import numpy as np
import scipy.stats as stats
from copy import deepcopy

# Class (semantic/perceptual)
# Category (color/shape)
# Features

def make_stim(feat_classes=2, feat_categories=(1, 2), feat_per_category=(10, 5)):
    feats = []
    for c in range(feat_classes):
        feats.append(np.random.normal(size=(feat_categories[c],
                                            feat_per_category[0])))
    return feats

def make_neural_responses(all_stim, noise=1):
    feat_classes = len(all_stim[0])
    for c in range(feat_classes):
        for i in range(len(all_stim)):
            all_stim[i][c] += np.random.normal(scale=noise, size=all_stim[i][c].shape)
    return all_stim

def get_RSMs(all_stim, just_cat=0, just_feats=3):
    feat_classes = len(all_stim[0])
    RSMs = []
    for c in range(feat_classes):
        if just_cat is not None:
            mat = [all_stim[i][c][just_cat][:just_feats] for i in range(len(all_stim))]
        else:
            mat = [all_stim[i][c].flatten() for i in range(len(all_stim))]
        # print(np.array(mat).shape)
        RSM = np.corrcoef(mat)
        RSMs.append(RSM)
    return RSMs

def do_RSA(model_RSMs, neural_RSMs):
    feat_classes = len(model_RSMs)
    for c in range(feat_classes):
        model_RSM = model_RSMs[c]
        neural_RSM = neural_RSMs[c]
        r, p = stats.pearsonr(model_RSM.flatten(), neural_RSM.flatten())
        print(f'Class {c}: {r=:.3f}, {p=:.3f}')

if __name__ == '__main__':

    for i in range(100):
        test = (np.random.randint(0, 1, size=(20, 20)) - 0.5) * 0.2
        test[np.diag_indices_from(test)] = 1
        sanity = np.random.multivariate_normal(np.zeros(20), cov=test, size=10,
                                               check_valid='raise')

    # print(sanity)
    quit()



    NUM_STIM = 1000
    FEAT_CLASSES = 2
    FEAT_CATEGORIES = (1, 2)
    FEAT_PER_CATEGORY = (100, 200)

    ALL_STIM = [make_stim(feat_per_category=FEAT_PER_CATEGORY)
                for _ in range(NUM_STIM)]
    ALL_NEURAL = make_neural_responses(deepcopy(ALL_STIM))
    MODEL_RSMs = get_RSMs(ALL_STIM)
    NEURAL_RSMs = get_RSMs(ALL_NEURAL)
    do_RSA(MODEL_RSMs, NEURAL_RSMs)
