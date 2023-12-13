from time import time

from nilearn import image

from utils import pickle_wrap
from scipy import io
from glob import glob
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
               'rock-climbing shoe': 'rockclimbing shoe', # maybe not the best
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
               'haircomb': 'comb', # object
               'soccerball': 'soccer ball', # object
               'McDonald\'s': 'fast food',
               #'ATM': 'ATM', # automatic teller machine
               'college quad': 'college campus',
               #'picnic blanket': 'blanket',
               }


def get_trial_info(sn, easy_override=False, ret=True):
    # ret may not be needed. added in 11/25/2025 but it wasnt needed
    ret_str = '_NoRet' if not ret else ''
    fp = fr'cache/trial_info/{sn}{ret_str}.pkl'
    df_sn = pickle_wrap(fp, lambda: get_trial_info_(sn, ret),
                        easy_override=easy_override,
                        verbose=False)
    return df_sn

def get_trial_info_(sn, ret=False):
    renamer = NAME_RENAMER

    obj_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/objects'
    obj_root2 = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/Enc_rerun/obj'
    scn_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/scenes'
    scn_root2 = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/Enc_rerun/scn'
    obj_root3 = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/ENC_rerun3/OBJ'
    scn_root3 = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/ENC_rerun3/SCN'
    cmb_root3 = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/ENC_rerun3/CMB'

    df_sn_as_l = []
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/ENC/S{sn}_run{run}.mat'
        mat_enc = io.loadmat(fp_bhv)


        for i in range(38):
            trial = i + 1
            try:
                glob_obj = fr'{obj_root}/Day2_Run{run}_Trial{trial}_*.nii'
                glob_obj = glob(glob_obj)
            # assert len(glob_obj) == 1, f'{sn}, trial: {trial}'
                fp_obj = glob_obj[0]
                glob_scn = fr'{scn_root}/Day2_Run{run}_Trial{trial}_*.nii'
                glob_scn = glob(glob_scn)
                # assert len(glob_scn) == 1
                fp_scn = glob_scn[0]
            except IndexError:
                fp_obj = None
                fp_scn = None

            if sn[0] == '2':
                fp_obj2 = None
            else:
                glob_obj2 = fr'{obj_root2}/ENC_sub{sn}_run{run}_trial{trial}_*.nii'
                glob_obj2 = glob(glob_obj2)
                assert len(glob_obj2) == 1
                fp_obj2 = glob_obj2[0]

            glob_obj3 = fr'{obj_root3}/ENC_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_obj3 = glob(glob_obj3)
            try:
                # assert len(glob_obj3) == 1
                fp_obj3 = glob_obj3[0]
            except IndexError:
                if sn == '224' and run == 3:
                    fp_obj3 = None
                elif sn == '234' and run == 1:
                    fp_obj3 = None
                else:
                    raise IndexError(f'{sn}, {run}, {trial}')


            if sn[0] == '2':
                fp_scn2 = None
            else:
                glob_scn2 = fr'{scn_root2}/ENC_sub{sn}_run{run}_trial{trial}_*.nii'
                glob_scn2 = glob(glob_scn2)
                assert len(glob_scn2) == 1
                fp_scn2 = glob_scn2[0]

            glob_scn3 = fr'{scn_root3}/ENC_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_scn3 = glob(glob_scn3)
            try:
                fp_scn3 = glob_scn3[0]
            except IndexError:
                if sn == '224' and run == 3:
                    fp_scn3 = None
                elif sn == '234' and run == 1:
                    fp_obj3 = None
                else:
                    raise IndexError(f'{sn}, {run}, {trial}')

            # glob_scn3 = fr'{scn_root3}/ENC_sub{sn}_run{run}_trial{trial}_*.nii'
            # glob_scn3 = glob(glob_scn3)
            # assert len(glob_scn3) == 1
            # fp_scn3 = glob_scn3[0]

            obj = mat_enc['pdata'][0][0][7][0][i][0]
            scene = mat_enc['pdata'][0][0][8][0][i][0]
            inc = mat_enc['pdata'][0][0][9][0][i][0][0]
            resp = mat_enc['pdata'][0][0][11][0][i][0][0]
            # if pd.isna(resp):
            #     perceived_inc = np.nan
            # elif resp == 1:
            #     perceived_inc = 1
            # elif resp == 4:
            #     perceived_inc = 3
            # else:
            #     perceived_inc = 2

            if obj in renamer:
                obj_rename = renamer[obj]
            else:
                obj_rename = obj
            if scene in renamer:
                scene_rename = renamer[scene]
            else:
                scene_rename = scene

            inc2str = {1: 'incongruent', 2: 'neutral', 3: 'congruent'}
            inc_str = inc2str[inc]
            per2str = {1: 'incongruent', 4: 'congruent'}


            trial_full = trial + 38 * (run - 1)
            per_inc14 = resp if resp in [1, 4] else np.nan
            per_inc14_str = np.nan if pd.isna(per_inc14) else per2str[per_inc14]



            d = {'sn': sn,
                 'enc_trial': trial_full,
                 'obj_trial': trial_full,
                 'scn_trial': trial_full,
                 'enc_run': run,
                 'obj_run': run,
                 'scn_run': run,
                 #'fp_fMRI': fp_obj,
                 'obj_fMRI': fp_obj,
                 'obj2_fMRI': fp_obj2,
                 'obj3_fMRI': fp_obj3,
                 'scn_fMRI': fp_scn,
                 'scn2_fMRI': fp_scn2,
                 'scn3_fMRI': fp_scn3,
                 # 'scn3_fMRI': fp_scn3,
                 'obj': obj,
                 'scene': scene,
                 'obj_rename': obj_rename,
                 'scene_rename': scene_rename,
                 'inc': inc,
                 'inc_str': inc_str,
                 'per_inc': resp, # higher (up to 4) = seen as congruent
                 'per_inc14': per_inc14,
                 'per_inc14_str': per_inc14_str,
                 # 'per_inc': perceived_inc,
                 # 'per_inc_str': per_inc_str,

                 }
            df_sn_as_l.append(d)
    df_sn = pd.DataFrame(df_sn_as_l)
    # add_onset_time(df_sn, sn)

    df_sn = include_BL(df_sn, sn)
    if ret:
        df_sn = include_conceptual(df_sn, sn)
        df_sn = include_vis(df_sn, sn)
    try:
        df_sn['hit_hit'] = df_sn['con_hit'] & df_sn['vis_hit']
        def f(row):
            if pd.isna(row['con_hit']) or pd.isna(row['vis_hit']):
                return np.nan
            else:
                return row['con_hit'] & row['vis_hit']
        df_sn['hit_hit_nan'] = df_sn.apply(f, axis=1)
    except KeyError as e:
        pass

    df_sn['hit_hit'] = df_sn['con_hit'] & df_sn['vis_hit']
    def f(row):
        if pd.isna(row['con_hit']) or pd.isna(row['vis_hit']):
            return np.nan
        else:
            return row['con_hit'] & row['vis_hit']
    df_sn['hit_hit_nan'] = df_sn.apply(f, axis=1)


    df_sn['inc_hit_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['hit_hit']) else
        f'{row["inc"]}{int(row["hit_hit"])}', axis=1)
    df_sn['inc_vis_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['vis_hit']) else
        f'{row["inc"]}{int(row["vis_hit"])}', axis=1)
    df_sn['inc_con_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['con_hit']) else
        f'{row["inc"]}{int(row["con_hit"])}', axis=1)
    # df_sn = prep_dif(df_sn, sn)
    return df_sn

def add_onset_time(df_sn, sn):
    trial_info = r'single_trial_conn/trial_info_all.csv'
    df_info = pd.read_csv(trial_info)
    df_info = df_info[df_info['ObjectTypeLabel_RCON'] == 'Old']
    df_info_sn = df_info[df_info['Subject'] == int(sn)]
    # df_info_sn = df_info_sn[df_info_sn['Run_BL'] == 1]
    # print(df_info_sn)
    obj_l = df_info_sn['Object'].values
    for name, stim, key in [('BL', 'Obj', 'bl'),
                            ('ENC', 'Obj', 'obj'),
                            ('ENC', 'Scene', 'scn'),
                            ('RCON', 'Obj', 'con'),
                            ('RVIS', 'Obj', 'vis')]:
        onset_col = f'Onset{stim}_{name}'
        onsets = df_info_sn[onset_col].values
        obj2onsets = dict(zip(obj_l, onsets))
        df_sn[f'{key}_onset'] = df_sn['obj'].map(obj2onsets)
        df_sn[f'{key}_onset_TR'] = 4 + df_sn[f'{key}_onset'] // 2# + 1
        # print(df_sn[f'{key}_onset_TR'].min())
        # print(df_sn[f'{key}_onset'].min())
        # quit()
        # add 1 at end to be ceil for int rounding
        try:
            df_sn[f'{key}_onset_TR'] = df_sn[f'{key}_onset_TR'].astype(int)
        except pd.errors.IntCastingNaNError:
            assert sn == '133' or sn == '138'# or sn == '212'
            # TODO: investigate 212...


def include_conceptual(df_sn, sn):
    conc_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}'
    obj2resp = {}
    obj2run = {}
    obj2old_new = {}
    obj2rt = {}
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2trial = {}
    # obj2run = {}
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/RET_con/S{sn}_run{run}_RC.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing CON behavioral data: {sn}, {run=}')
            continue


        has_missing_fMRI = False
        for i in range(48):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            obj2run[obj] = run
            obj2trial[obj] = trial + (run - 1) * 48
            old_new = mat_enc['pdata'][0][0][8][0][i][0][0]
            assert old_new in [0, 1], f'Old new not 0 or 1: {old_new=}'
            obj2old_new[obj] =' new' if old_new else 'old'
            resp = mat_enc['pdata'][0][0][9][0][i][0]
            # print(f'{run}, trial: {i} ({obj}) | {resp=}')
            obj2resp[obj] = None if pd.isna(resp) else int(resp)
            obj2rt[obj] = mat_enc['pdata'][0][0][10][0][i][0]
            glob_conc = fr'{conc_root}/all_CONruns_sorted/Day3Conceptual_Run{run}_Trial{trial}_*.nii'
            glob_conc = glob(glob_conc)
            if len(glob_conc) < 1:
                #print(f'No glob_conc ({sn}): {glob_conc=}')
                obj2fp[obj] = np.nan
            else:
                obj2fp[obj] = glob_conc[0]

            if sn[0] == '2':
                obj2fp2[obj] = None
            else:
                glob_conc2 =  fr'{conc_root}/CON_rerun/RCON_sub{sn}_run{run}_trial{trial}_*.nii'
                glob_conc2 = glob(glob_conc2)
                if len(glob_conc2):
                    assert len(glob_conc2) == 1
                    obj2fp2[obj] = glob_conc2[0]
                else:
                    obj2fp2[obj] = None

            glob_conc3 = fr'{conc_root}/CON_rerun3/CONC/RCON_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_conc3 = glob(glob_conc3)
            if len(glob_conc3):
                assert len(glob_conc3) == 1
                obj2fp3[obj] = glob_conc3[0]
            else:
                if len(glob_conc) < 1:
                    has_missing_fMRI = True

                #break
            # assert len(glob_conc) == 1, f'{len(glob_conc)=}'
            # obj2trial
        # if has_missing_fMRI:
            # print(f'Missing conceptual fMRI data: {sn}, {run=}')
    else:
        df_sn['con_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['con_hit'] = df_sn['con_resp'].apply(
            lambda x: np.nan if pd.isna(x) else x >= 3)
        df_sn['con_fMRI'] = df_sn['obj'].map(obj2fp)
        df_sn['con2_fMRI'] = df_sn['obj'].map(obj2fp2)
        df_sn['con3_fMRI'] = df_sn['obj'].map(obj2fp3)
        df_sn['con_run'] = df_sn['obj'].map(obj2run)
        df_sn['con_trial'] = df_sn['obj'].map(obj2trial)

    return df_sn

def include_vis(df_sn, sn):
    # TODO: investigate why 138 is missing run3 visual retrieval
    # Figure out the trial breakdown
    vis_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}'
    obj2resp = {}
    obj2type = {} # unused
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2rt = {}
    obj2run = {}
    obj2trial = {}

    for run in range(1, 4):
        fp_bhv = fr'behavFiles/RET_vis/S{sn}_run{run}_RV.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing VIS behavioral data: {sn}, {run=}')
            continue
        if sn == '213' and run == 2: # not recorded
            continue
        for i in range(42):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            obj2run[obj] = run
            obj2trial[obj] = trial + 42 * (run - 1)
            resp = mat_enc['pdata'][0][0][8][0][i][0]
            resp = 'old' if resp == 3 else \
                   'similar' if resp == 2 else \
                   'new' if resp == 1 else np.nan
            obj2resp[obj] = resp
            obj2rt[obj] = mat_enc['pdata'][0][0][9][0][i][0]
            old_similar_new = mat_enc['pdata'][0][0][11][0][i][0][0]
            # TODO: For visual retrieval, confirm that 0 = old, 1 = similar, 2 = new
            old_similar_new = 'old' if old_similar_new == 0 else \
                'similar' if old_similar_new == 1 else 'new'
            obj2type[obj] = old_similar_new
            glob_vic = fr'{vis_root}/all_VISruns_sorted/Day3Visual_Run{run}_Trial{trial}_*.nii'
            glob_vic = glob(glob_vic)
            if len(glob_vic) < 1:
                obj2fp[obj] = None
            else:
                obj2fp[obj] = glob_vic[0]

            if sn == '138': # has original preprocessing but no rerun betas?
                continue

            if sn[0] == '2':
                obj2fp2[obj] = None
            else:
                glob_vic2 = fr'{vis_root}/VIS_rerun/VIS/RVIS_sub{sn}_run{run + 3}_trial{trial}_*.nii'
                glob_vic2 = glob(glob_vic2)
                if len(glob_vic2):
                    obj2fp2[obj] = glob_vic2[0]
                else:
                    obj2fp2[obj] = None

            glob_vic3 = fr'{vis_root}/VIS_rerun3/VIS/RVIS_sub{sn}_run{run + 3}_trial{trial}_*.nii'
            glob_vic3 = glob(glob_vic3)
            if len(glob_vic3):
                obj2fp3[obj] = glob_vic3[0]
            else:
                obj2fp3[obj] = None

    else:
        df_sn['vis_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['vis_type'] = df_sn['obj'].map(obj2type)
        f = lambda row: np.nan if pd.isna(row['vis_resp']) else \
            row['vis_resp'] == row['vis_type']
        df_sn['vis_hit'] = df_sn.apply(f, axis=1)
        df_sn['vis_fMRI'] = df_sn['obj'].map(obj2fp)
        df_sn['vis2_fMRI'] = df_sn['obj'].map(obj2fp2)
        df_sn['vis3_fMRI'] = df_sn['obj'].map(obj2fp3)
        df_sn['vis_run'] = df_sn['obj'].map(obj2run)
        df_sn['vis_trial'] = df_sn['obj'].map(obj2trial)
        # if sn not in get_bad_sns(ret=True):
        #     assert len(df_sn['vis2_fMRI'].value_counts()) == 114, \
        #         f'Missing vis fMRI fp: {len(df_sn["vis2_fMRI"].value_counts())}'


    return df_sn

def do_BL_move(bl_root, run, trial):
    glob_BL_pre = fr'{bl_root}/Day1_Run{run}_Trial{trial}_*.nii'
    glob_BL_pre = glob(glob_BL_pre)
    if len(glob_BL_pre) > 1:
        print(f'Bad more than one pre: {glob_BL_pre=}')
        quit()
    elif len(glob_BL_pre) == 1:
        import shutil
        fp_BL_pre = Path(glob_BL_pre[0])
        dir_BL_post = fp_BL_pre.parent.joinpath('all_BLruns_sorted')
        dir_BL_post.mkdir(exist_ok=True)
        fp_BL_post = dir_BL_post.joinpath(fp_BL_pre.name)
        shutil.move(fp_BL_pre, fp_BL_post)

def include_BL(df_sn, sn):
    bl_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}'
    obj2resp = {}
    obj2run = {}
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2trial = {}
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/bl/S{sn}_run{run}.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing BL behavioral data: {sn}, {run=}')
            continue

        # if sn == '135':
        #     # 135 has no rerun data
        #     continue

        for i in range(38):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            obj2run[obj] = run
            obj2trial[obj] = trial + 38 * (run - 1)
            resp = mat_enc['pdata'][0][0][8][0][i][0]
            obj2resp[obj] = resp
            do_BL_move(bl_root, run, trial)

            glob_BL = fr'{bl_root}/all_BLruns_sorted/Day1_Run{run}_Trial{trial}_*.nii'
            glob_BL = glob(glob_BL)
            if len(glob_BL):
                obj2fp[obj] = glob_BL[0]

            if sn[0] == '2':
                obj2fp2[obj] = None
            else:
                glob_BL2 = fr'{bl_root}/BL_rerun/bl/BL_sub{sn}_run{run}_trial{trial}_*.nii'
                glob_BL2 = glob(glob_BL2)
                assert len(glob_BL2) == 1 or len(glob_BL) == 1, f'Bad bl missing'
                if len(glob_BL2):
                    obj2fp2[obj] = glob_BL2[0]
                else:
                    obj2fp2[obj] = None

            glob_BL3 = fr'{bl_root}/BL_rerun3/BL/BL_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_BL3 = glob(glob_BL3)
            assert len(glob_BL3) == 1 or len(glob_BL) == 1, f'Bad bl3 missing'
            if len(glob_BL3):
                obj2fp3[obj] = glob_BL3[0]
            else:
                obj2fp3[obj] = None

    else:
        df_sn['bl_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['bl_fMRI'] = df_sn['obj'].map(obj2fp)
        df_sn['bl2_fMRI'] = df_sn['obj'].map(obj2fp2)
        df_sn['bl3_fMRI'] = df_sn['obj'].map(obj2fp3)
        df_sn['bl_run'] = df_sn['obj'].map(obj2run)
        df_sn['bl_trial'] = df_sn['obj'].map(obj2trial)


        # assert pd.isna(df_sn['bl2_fMRI']).sum() == 0 or sn == '135' or \
        #        sn[0] == '2', 'Missing bl fMRI fp unneeded'
        # assert len(df_sn['bl_fMRI'].value_counts()) == 114, 'Missing bl fMRI fp'
    return df_sn


def prep_dif(df, sn):
    t = time()
    dif_root = Path(fr'Day2EncSingleTrialModellingLSS_sorted\{sn}')
    key_pairs = [('bl', 'obj'), ('bl', 'vis'), ('obj', 'vis')]
    for key0, key1 in key_pairs:
        fp_key0 = f'{key0}_fMRI'
        fp_key1 = f'{key1}_fMRI'
        dif_fps = []
        for trial, run, fp0, fp1 in zip(df['enc_trial'], df['run'],
                                        df[fp_key0], df[fp_key1]):
            if pd.isna(fp0) or pd.isna(fp1):
                dif_fps.append(None)
                continue
            dir_out = dif_root.joinpath(fr'dif_{key0}_{key1}')
            dir_out.mkdir(exist_ok=True)
            # fp_out = dir_out.joinpath(f'EncTrial{trial}.nii')
            fp_out = dir_out.joinpath(f'Trial{trial}_Run{run}.nii')
            dif_fps.append(str(fp_out))
            if fp_out.is_file():
                continue
                # os.remove(fp_out)
                # continue
            fp_out = str(fp_out) # nilearn errors with pathlib Paths
            img0 = image.load_img(fp0)
            img1 = image.load_img(fp1)
            dif_img = image.math_img('img0 - img1', img0=img0, img1=img1)
            dif_img.to_filename(str(fp_out))
        df[f'dif_{key0}-{key1}'] = dif_fps
    print(f'Prepped difs for {sn} in {time() - t:.2f}s')
    return df

