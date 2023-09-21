import numpy as np
import pandas as pd


def get_stim_RDM(df_sn, d_vecs, obj_only=False, scene_only=False,
                 dif=False, prod=False, take_abs=False, add=False):
    # return get_stim_RDM_lifu(df_sn)
    assert obj_only or scene_only or dif or prod or add, \
        'Must specify one of obj_only, scene_only, dif, prod, add'
    vec_size = len(d_vecs[df_sn['obj'].iloc[0]])
    vecs_obj = np.empty((len(df_sn['obj']), vec_size))
    vecs_scene = np.empty((len(df_sn['obj']), vec_size))
    for i, (obj, scene, obj_rename, scene_rename) in enumerate(zip(df_sn['obj'],
            df_sn['scene'], df_sn['obj_rename'], df_sn['scene_rename'])):
        vec_obj = d_vecs[obj]
        vec_scene = d_vecs[scene]
        vecs_obj[i, :] = vec_obj
        vecs_scene[i, :] = vec_scene
    if obj_only:
        all_vecs = vecs_obj
    elif scene_only:
        all_vecs = vecs_scene
    elif dif:
        all_vecs = vecs_obj - vecs_scene
    elif prod:
        all_vecs = vecs_obj * vecs_scene
    elif add:
        all_vecs = vecs_obj + vecs_scene
    else:
        raise ValueError('How?? This error should\'ve been caught by assert.')
    all_vecs = np.abs(all_vecs) if take_abs else all_vecs
    M = np.mean(all_vecs)
    print(f'{M=}')
    RDM_stim = np.corrcoef(all_vecs)
    pd.DataFrame(RDM_stim).to_csv('RDM_stim_mine.csv')
    return RDM_stim
