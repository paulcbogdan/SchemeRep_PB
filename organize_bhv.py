import os.path
from collections import defaultdict

from scipy import io
from glob import glob
from pprint import pprint
import pandas as pd
import numpy as np
from pathlib import Path

# TODO: measure where congruent is more correlated object x scene

NAME_RENAMER = {'inside of a car': 'car',
               'surfing board': 'surfboard', # object
               'tropical volcano': 'volcano',
               'coffee shop': 'cafe',
               #'religious statue': 'statuette',
               #'oversize tire': 'tire',
               'restroom stall': 'bathroom',
               'front porch': 'porch',
               'ice stadium': 'hockey rink',
               #'laundry hamper': 'laundry basket',
               'dumb bell': 'dumbbell',
               # 'rock-climbing shoe': 'climbing shoe',
               'rock-climbing shoe': 'rock climbing shoe', # maybe not the best
               # 'book bag': 'backpack',
               'sea gull': 'seagull', # object
               'hair salon': 'salon',
               #'movie theater': 'theater',
               'snowy mountains': 'mountain',
               #'dining chair': 'chair',
               #'dining table': 'table',
               #'police baton': 'police baton',
               'grocery store': 'supermarket',
               'apartment complex': 'apartment',
               'concert hall': 'orchestra',
               'display cabinet': 'cabinet',
               #'potted plant': 'plant',
               #'construction helmet': 'hard hat',
               #'game token': 'game token',
               'Eiffel Tower': 'paris landmark',
               #'binder clip': 'binder clip',
               'office space': 'office',
               'haircomb': 'hair comb', # object
               'soccerball': 'soccer ball', # object
               'McDonald\'s': 'fast food',
               #'ATM': 'ATM', # automatic teller machine
               'college quad': 'college campus',
               #'picnic blanket': 'blanket',
               }



def get_trial_info(sn, ret=False):
    renamer = NAME_RENAMER

    obj_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/objects'
    scn_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/scenes'
    df_sn_as_l = []
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/ENC/S{sn}_run{run}.mat'
        mat_enc = io.loadmat(fp_bhv)
        for i in range(38):
            trial = i + 1
            glob_obj = fr'{obj_root}/Day2_Run{run}_Trial{trial}_*.nii'
            glob_obj = glob(glob_obj)
            assert len(glob_obj) == 1
            fp_obj = glob_obj[0]

            glob_scn = fr'{scn_root}/Day2_Run{run}_Trial{trial}_*.nii'
            glob_scn = glob(glob_scn)
            assert len(glob_scn) == 1
            fp_scn = glob_scn[0]

            obj = mat_enc['pdata'][0][0][7][0][i][0]
            scene = mat_enc['pdata'][0][0][8][0][i][0]
            CIN = mat_enc['pdata'][0][0][9][0][i][0][0]
            resp = mat_enc['pdata'][0][0][11][0][i][0][0]

            if obj in renamer:
                obj_rename = renamer[obj]
            else:
                obj_rename = obj
            if scene in renamer:
                scene_rename = renamer[scene]
            else:
                scene_rename = scene

            d = {'trial': trial,
                 'run': run,
                 #'fp_fMRI': fp_obj,
                 'obj_fMRI': fp_obj,
                 'scn_fMRI': fp_scn,
                 'obj': obj,
                 'scene': scene,
                 'obj_rename': obj_rename,
                 'scene_rename': scene_rename,
                 'CIN': CIN,
                 'perceived_con': resp # higher (up to 4) = seen as congruent
                 }
            df_sn_as_l.append(d)
    df_sn = pd.DataFrame(df_sn_as_l)
    df_sn = process_conc_retrieval(df_sn, sn)
    df_sn = process_vis_retrieval(df_sn, sn)
    return df_sn

def process_conc_retrieval(df_sn, sn):
    conc_root = fr'Day2EncSingleTrialModellingLSS_sorted\{sn}\all_CONruns_sorted'
    obj2resp = {}
    obj2old_new = {}
    obj2rt = {}
    obj2conc = {}
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/RET_con/S{sn}_run{run}_RC.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            continue
        for i in range(48):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            old_new = mat_enc['pdata'][0][0][8][0][i][0][0]
            assert old_new in [0, 1], f'Old new not 0 or 1: {old_new=}'
            obj2old_new[obj] =' new' if old_new else 'old'
            resp = mat_enc['pdata'][0][0][9][0][i][0]
            obj2resp[obj] = None if pd.isna(resp) else int(resp)
            obj2rt[obj] = mat_enc['pdata'][0][0][10][0][i][0]
            glob_conc = fr'{conc_root}/Day3Conceptual_Run{run}_Trial{trial}_*.nii'
            glob_conc = glob(glob_conc)
            if len(glob_conc) < 1:
                print(f'No glob_conc ({sn}): {glob_conc=}')
                break
            # assert len(glob_conc) == 1, f'{len(glob_conc)=}'
            obj2conc[obj] = glob_conc[0]
    else:
        df_sn['con_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['hit_bool'] = df_sn['con_resp'].apply(
            lambda x: np.nan if pd.isna(x) else x >= 3)
        df_sn['con_fMRI'] = df_sn['obj'].map(obj2conc)
        # for x in df_sn['con_fMRI']:
        #     exists = os.path.isfile(x)
        #     if not exists:
        #         print(f'BAD: {x}')
    return df_sn

def process_vis_retrieval(df_sn, sn):
    # TODO: investigate why 138 is missing run3 visual retrieval
    # Figure out the trial breakdown
    vis_root = fr'Day2EncSingleTrialModellingLSS_sorted\{sn}\all_VISruns_sorted'

    obj2resp = {}
    obj2type = {} # unused
    obj2vis = {}
    obj2rt = {}
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/RET_vis/S{sn}_run{run}_RV.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            continue
        for i in range(42):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            resp = mat_enc['pdata'][0][0][8][0][i][0]
            resp = 'old' if resp == 3 else \
                   'similar' if resp == 2 else \
                   'new' if resp == 1 else np.nan
            # print(resp)
            obj2resp[obj] = resp
            # obj2resp[obj] = None if pd.isna(resp) else int(resp)
            obj2rt[obj] = mat_enc['pdata'][0][0][9][0][i][0]
            old_similar_new = mat_enc['pdata'][0][0][11][0][i][0][0]
            # TODO: For visual retrieval, confirm that 0 = old, 1 = similar, 2 = new
            old_similar_new = 'old' if old_similar_new == 0 else \
                'similar' if old_similar_new == 1 else 'new'
            obj2type[obj] = old_similar_new
            # print(f'{obj=}, {old_similar_new=}')
            glob_vic = fr'{vis_root}/Day3Visual_Run{run}_Trial{trial}_*.nii'
            glob_vic = glob(glob_vic)
            if len(glob_vic) < 1:
                obj2vis[obj] = None
                # print(f'No visual glob for {obj=}, {old_similar_new=}')
                # break
                continue
            assert len(glob_vic) == 1
            obj2vis[obj] = glob_vic[0]
    else:
        df_sn['vis_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['vis_type'] = df_sn['obj'].map(obj2type)
        f = lambda row: np.nan if pd.isna(row['vis_resp']) else \
            row['vis_resp'] == row['vis_type']
        df_sn['hit_bool'] = df_sn.apply(f, axis=1)
        df_sn['vis_fMRI'] = df_sn['obj'].map(obj2vis)
        # goods = 1
        # for x in df_sn['vis_fMRI']:
        #     if pd.isna(x):
        #         print('NaN')
        #         continue
        #     exists = os.path.isfile(x)
        #     if not exists:
        #         print(f'BAD: {x}')

    # pd.set_option('display.max_columns', None)
    # print(df_sn['vis_resp'].value_counts())
    # print(df_sn['hit_bool'].value_counts(dropna=False))
    # print(dict(df_sn['vis_fMRI'].value_counts(dropna=False)))
    # quit()
    return df_sn

def get_all_sns(ret=False):
    age2sn = defaultdict(list)
    bad_sns = {'126', '131',
               '201', '224', '231', '232', '233', '234', '235'}
    if ret:
        bad_sns.add('116')
        bad_sns.add('125')
        bad_sns.add('133')
        bad_sns.add('138') # Has con but not vis

        bad_sns.add('213')
        bad_sns.add('215')
    for age in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{age}*_run1.mat'
        fps = glob(bhv_root)
        for fp in fps:
            sn = Path(fp).stem
            sn_no_S = sn[1:]
            sn = sn_no_S.replace('_run1', '')
            # sn = fn.replace(r'behavFiles/ENC/S', '').replace('_run1.mat', '')
            if sn in bad_sns:
                continue
            age2sn[age].append(sn)
    age2sn['healthy'] = age2sn[1] + age2sn[2]
    return age2sn

if __name__ == '__main__':
    age2sn = get_all_sns(ret=True)
    for sn in age2sn[1]:
        print(sn)
        get_trial_info(sn)