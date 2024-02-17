from utils import pickle_wrap
from collections import defaultdict

import numpy as np

from atlas_utils import get_BN_and_resample, get_combined_BNA
from organize_bhv import get_trial_info
from org_sns import get_sns
from nilearn import image

from stim import get_stim_RDM, get_semantic_vectors, get_DNN_vecs
from utils import stdize, nan_ar, defaultdict_to_dict, pb_outer_double_multi
import utils
import scipy.stats as stats

from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def plot_RDMs(semantic=False, early=True, cin=None):
    df_sn = get_trial_info('102')

    if cin:
        df_sn = df_sn[df_sn['CIN'] == cin]
    # df_sn = df_sn.sort_values('CIN')

    if semantic:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(early=early, PCA=True)

    RDM_stims = {'obj': get_stim_RDM(df_sn, d_vecs, obj_only=True),
                 'obj_abs': get_stim_RDM(df_sn, d_vecs, obj_only=True, take_abs=True),
                 'scn': get_stim_RDM(df_sn, d_vecs, scene_only=True),
                 'scn_abs': get_stim_RDM(df_sn, d_vecs, scene_only=True, take_abs=True),
                 'dif': get_stim_RDM(df_sn, d_vecs, dif=True),
                 'dif_abs': get_stim_RDM(df_sn, d_vecs, dif=True, take_abs=True),
                 'prd': get_stim_RDM(df_sn, d_vecs, prod=True),
                 'prd_abs': get_stim_RDM(df_sn, d_vecs, prod=True, take_abs=True),
                 'add': get_stim_RDM(df_sn, d_vecs, add=True),
                 'add_abs': get_stim_RDM(df_sn, d_vecs, add=True, take_abs=True),
                 }

    for key, RDM in RDM_stims.items():
        if 'scn' not in key:
            continue
        RDM[np.diag_indices_from(RDM)] = np.nan
        M_val = np.nanmedian(RDM)
        print(f'{key=} [{cin=}], {M_val=:.4f}')
        plt.imshow(RDM, cmap='turbo')

        key_str = utils.get_RSA_name(key).replace(' RSA', '')
        if semantic:
            rsa_str = 'word2vec'
        else:
            rsa_str = '1st-layer DNN' if early else 'late-layer DNN'
        title = f'{rsa_str}, {key_str} RSM. Median = {M_val:.3f}'
        plt.title(title)
        plt.ylabel('Stimulus A')
        plt.xlabel('Stimulus B')
        vmin = np.nanquantile(RDM, .975)
        vmax = np.nanquantile(RDM, .025)
        # vabs = max(abs(vmin), abs(vmax))
        plt.clim(vmin, vmax)

        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    plot_RDMs()