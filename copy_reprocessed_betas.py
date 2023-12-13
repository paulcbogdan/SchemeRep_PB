from pathlib import Path
import os
from tqdm import tqdm
import shutil


if __name__ == '__main__':

    dir_out = r'Day2EncSingleTrialModellingLSS_sorted'

    scans = [
        # 'BL__full__regBPtrue',
        # 'RCON__full__regBPtrue',
        # 'RVIS__full__regBPtrue',
        # 'ENC__full__regBPtrue__scene_obj_separate',
        # 'ENC__full__regBPtrue__scene_obj_separate',
        'ENC__full__regBPtrue__scene_obj_combined'
    ]

    details = [
        # ('b_Object.nii', 'BL_rerun3', 'BL'),
        # ('b_Object.nii', 'CON_rerun3', 'CONC'),
        # ('b_Object.nii', 'VIS_rerun3', 'VIS'),
        # ('b_Object.nii', 'ENC_rerun3', 'OBJ'),
        # ('b_Scene.nii', 'ENC_rerun3', 'SCN'),
        ('b_SceneObject.nii', 'ENC_rerun3', 'CMB')
    ]


    for scan, detail in zip(scans, details):
        enc_dir = fr'Y:\SchemRep.01\Scripts\SingleTrialModel_SH\model_output\{scan}'
        enc_dir = Path(enc_dir)
        g = enc_dir.glob('*')
        g = [sn.name for sn in g]# if sn.name[:1] == '2']
        # g = ['234']
        # print(list(g))
        # quit()
        # g = g[::-1]
        # g = [sn for sn in g if sn[:3] == '129' or sn[:3] == '128' or sn[:3] == '127' or sn[:3] == '126']
        for sn in tqdm(g, desc=f'Looping sn: {scan}'):
            sn_dir = enc_dir.joinpath(sn)
            trials = sn_dir.glob('*')
            trials = (trial for trial in trials if trial.is_dir() )
            for trial in tqdm(trials, desc=f'Looping subj: {sn}'):
                fp_obj = trial.joinpath(detail[0])
                fn_obj_out = trial.name + '.nii'
                dir_enc_out = Path(dir_out).joinpath(sn).joinpath(detail[1])
                fp_obj_out = dir_enc_out.joinpath(detail[2]).joinpath(fn_obj_out)
                if fp_obj_out.exists() and os.path.getsize(str(fp_obj_out)) > 4e6:
                    continue
                fp_obj_out.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(fp_obj, fp_obj_out)
                continue

                fp_obj = trial.joinpath('b_Object.nii')
                fn_obj_out = trial.name + '.nii'
                dir_enc_out = Path(dir_out).joinpath(sn).joinpath('Enc_rerun')
                fp_obj_out = dir_enc_out.joinpath('obj').joinpath(fn_obj_out)
                if fp_obj_out.exists():
                    continue
                fp_obj_out.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(fp_obj, fp_obj_out)


                fp_scn = trial.joinpath('b_Object.nii')
                dir_enc_out = Path(dir_out).joinpath(sn).joinpath('Enc_rerun')
                fp_scn_out = dir_enc_out.joinpath('scn').joinpath(fn_obj_out)
                if fp_scn_out.exists():
                    continue
                fp_scn_out.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(fp_scn, fp_scn_out)