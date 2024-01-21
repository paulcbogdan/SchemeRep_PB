import numpy as np
from scipy import stats

from organize_bhv import get_trial_info
from stim import get_semantic_vectors, get_DNN_vecs, get_stim_RDM

if __name__ == '__main__':
    SEMANTIC = False
    if SEMANTIC:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    df_sn = get_trial_info('102', easy_override=False)

    for inc in [1, 2, 3]:
        df_sn_ = df_sn[df_sn['inc'] == inc]

        obj_RDM = get_stim_RDM(df_sn_, d_vecs, obj_only=True)
        obj_RDM_flat = obj_RDM[np.tril_indices_from(obj_RDM, k=-1)]
        scn_RDM = get_stim_RDM(df_sn_, d_vecs, scene_only=True)
        scn_RDM_flat = scn_RDM[np.tril_indices_from(scn_RDM, k=-1)]
        r, p = stats.spearmanr(obj_RDM_flat, scn_RDM_flat)

        print(f'{inc=} | {r=:.3f}, {p=:.3f}')








