from collections import defaultdict

import numpy as np
from pickle_wrap import pickle_wrap

from DNN_vectors import get_DNN_vecs, get_img_fns
# from analyze_conn import get_age_str, get_cin_str
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

from fMRI_analysis import get_atlas_resampled, get_all_sns
from organize_bhv import get_trial_info
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting



def get_FC(age, cin):
    atlas = get_atlas_resampled()
    masker = NiftiLabelsMasker(labels_img=atlas['maps'], standardize=True)
    age2sn = get_all_sns()
    mats = []
    for sn in age2sn[age]:
        df_sn = get_trial_info(sn)
        if cin is None:
            df_cond = df_sn
        else:
            df_cond = df_sn[df_sn['CIN'] == cin]

        img = image.load_img(df_cond['fp_fMRI'])
        # confounds = image.high_variance_confounds(img)
        time_series = masker.fit_transform(img)
                                           # confounds=confounds)
        correlation_measure = ConnectivityMeasure(
            kind="correlation",
        )
        correlation_matrix = correlation_measure.fit_transform(
            [time_series])[0]
        cin_ = 'all' if cin is None else cin
        mats.append(correlation_matrix)
    return np.array(mats)
            # plotting.plot_matrix(
            #     correlation_matrix, labels=atlas['labels'], colorbar=True, vmax=0.8, vmin=-0.8
            # )
            # plt.show()
            # quit()

if __name__ == '__main__':
    pass
    # for AGE in ['healthy', 1, 2]:
    #     for CIN in [None, 1, 2, 3]:
    #         age_str = get_age_str(AGE)
    #         cin_str = get_cin_str(CIN)
    #     fp_out = fr'cache/fCon_{age_str}{cin_str}.pkl'
