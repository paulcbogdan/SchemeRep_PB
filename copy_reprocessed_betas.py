from pathlib import Path
import os
from tqdm import tqdm
import shutil


if __name__ == '__main__':
    fp_in = r'/Volumes/Cabeza/SchemRep.01/Scripts/SingleTrialModel_SH/model_output/ENC__full__regBPtrue__scene_obj_separate/103/ENC_sub103_run1_trial1_subset3_pairID29/b_ButtonPress.nii'
    fp_in = Path(fp_in)
    test = fp_in.exists()
    print(test)

    dir_out = r'Day2EncSingleTrialModellingLSS_sorted'

    scans = ['BL__full__regBPtrue',
             'RCON__full__regBPtrue',
             'RVIS__full__regBPtrue',
             ]#'ENC__full__regBPtrue__scene_obj_separate',]

    details = [('b_Object.nii', 'BL_rerun', 'bl'),
               ('b_Object.nii', 'CON_rerun', 'CON'),
               ('b_Object.nii', 'VIS_rerun', 'VIS'),]
    # details = [
    #            ('b_Object.nii', 'VIS_rerun', 'VIS'),]

    # TODO: redo visual
    # TODO: is 135 missing bl?

    scans = ['ENC__full__regBPtrue__scene_obj_separate',
             ]
    details = [('b_Scene.nii', 'ENC_rerun', 'scn')]
    scans = ['RVIS__full__regBPtrue',]
    details = [('b_Object.nii', 'VIS_rerun', 'VIS')]

    # enc_dir = r'/Volumes/Cabeza/SchemRep.01/Scripts/SingleTrialModel_SH/model_output/ENC__full__regBPtrue__scene_obj_separate'


    for scan, detail in zip(scans, details):
        enc_dir = fr'/Volumes/Cabeza/SchemRep.01/Scripts/SingleTrialModel_SH/model_output/{scan}'
        enc_dir = Path(enc_dir)
        g = enc_dir.glob('*')
        # for sn in g:
        #     print(sn.name)
        g = [sn.name for sn in g if sn.name[:1] == '1']
        # print(g)
        # g = [sn for sn in g if sn[:3] == '129' or sn[:3] == '128' or sn[:3] == '127' or sn[:3] == '126']
        for sn in tqdm(g, desc='Looping sn'):
            sn_dir = enc_dir.joinpath(sn)
            trials = sn_dir.glob('*')
            trials = (trial for trial in trials if trial.is_dir() )
            for trial in tqdm(trials, desc=f'Looping subj: {sn}'):

                fp_obj = trial.joinpath(detail[0])
                fn_obj_out = trial.name + '.nii'
                dir_enc_out = Path(dir_out).joinpath(sn).joinpath(detail[1])
                fp_obj_out = dir_enc_out.joinpath(detail[2]).joinpath(fn_obj_out)
                # print(f'{os.path.getsize(str(fp_obj_out))=}')
                # quit()
                # if fp_obj_out.exists() and os.path.getsize(str(fp_obj_out)) > 4e6:
                #     continue
                fp_obj_out.parent.mkdir(exist_ok=True, parents=True)
                os.remove(fp_obj_out)
                print(f'{fp_obj_out=}')
                shutil.copyfile(fp_obj, fp_obj_out)
                quit()

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

            # fp_scn = trial.joinpath('b_Scene.nii')
            # fn_obj_out = fn_obj_out.replace('trial', 'Trial')


            # print(fn_obj_out)
            # print(fp_obj.exists())
            # print(fp_scn.exists())
            # quit()