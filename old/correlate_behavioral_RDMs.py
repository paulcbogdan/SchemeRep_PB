import pandas as pd
import numpy as np
from scipy import stats

from org_sns import get_sns
from organize_bhv import get_trial_info
from stim import get_semantic_vectors, get_DNN_vecs, get_stim_RDM

def vec_x_vec(l_obj, l_scn):
    z_l = []
    l_obj = np.array(l_obj)
    # print(np.mean(l_obj, axis=0)[None, :].shape)
    # quit()
    # print(l_obj)
    # l_obj -= np.mean(l_obj, axis=0)[None, :]
    # print(np.mean(l_obj, axis=0))
    # # print(l_obj)
    # quit()
    # l_scn = np.array(l_scn)
    # l_scn -= np.mean(l_scn, axis=0)[None, :]
    for v0, v1 in zip(l_obj, l_scn):
        # print(v0)
        # quit()
        r, p = stats.spearmanr(v0, v1)
        z = np.arctanh(r)
        z_l.append(z)
        # print(f'{r=:.3f}, {p=:.3f}')
    mean_z = np.mean(z_l)
    SD_z = np.std(z_l)
    SE_z = SD_z/np.sqrt(len(z_l))
    print(f'{mean_z=:.3f} [{SE_z=:.3f}], {SD_z=:.3f}')

if __name__ == '__main__':
    SEMANTIC = True
    if SEMANTIC:
        d_vecs = get_semantic_vectors()
    else:
        d_vecs = get_DNN_vecs(DNN_layer=2, PCA=True, PCA_obj=True)
    sns = get_sns('obj7_fMRI')[1]

    df_l = []
    for sn in sns:
        df_sn = get_trial_info(sn, easy_override=False)
        df_l.append(df_sn)
    df = pd.concat(df_l)
    df['scn_obj'] = df.apply(lambda row: (row['obj'], row['scene']), axis=1)
    df.drop_duplicates(subset=['scn_obj'], inplace=True)

    for inc in [1, 2, 3]:
        rs = []
        df_cond = df[df['inc'] == inc]
        obj_RDM = get_stim_RDM(df_cond, d_vecs, obj_only=True)
        obj_RDM_flat = obj_RDM[np.tril_indices_from(obj_RDM, k=-1)]
        scn_RDM = get_stim_RDM(df_cond, d_vecs, scene_only=True)
        scn_RDM_flat = scn_RDM[np.tril_indices_from(scn_RDM, k=-1)]
        r, p = stats.spearmanr(obj_RDM_flat, scn_RDM_flat)
        print(f'{inc=} | {r=:.3f}, {p=:.3f}')

        objs = df_cond['obj']
        obj_vecs = [d_vecs[obj] for obj in objs]
        scns = df_cond['scene']
        scn_vecs = [d_vecs[scn] for scn in scns]
        vec_x_vec(obj_vecs, scn_vecs)



