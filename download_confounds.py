from pathlib import Path
import os
from tqdm import tqdm
import shutil

from org_sns import get_sns

if __name__ == '__main__':
    root = r'Z:\Cabeza\SchemRep.01\Data\fMRIprep_by_subject_out'
    dir_out = r'cache\confounds'
    sn_dirs = [sn for sn in Path(root).glob('sub-*') if sn.is_dir()]
    print(sn_dirs)
    age2sns = get_sns()
    good_sns = age2sns[1] + age2sns[2]
    for sn_dir in tqdm(sn_dirs, desc='Downloading BOLD'):
        sn = sn_dir.name.split('-')[1]
        if sn not in good_sns:
            continue
        print(sn)
        did_rs = False
        for sess in range(1, 4):
            sess_dir = sn_dir / f'ses-{sess}'
            if not sess_dir.exists():
                continue
            num_runs = 6 if sess == 3 else 3
            for run in range(1, num_runs + 1):
                if sess == 1:
                    name = 'BL'
                elif sess == 2:
                    name = 'ENC'
                elif sess == 3:
                    if run < 4:
                        name = 'RCON'
                    else:
                        name = 'RVIS'
                else:
                    raise ValueError
                if name != 'ENC': continue

                fp_fMRI = (sess_dir / 'func' /
                           f'sub-{sn}_ses-{sess}_task-{name}_run-0{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
                dir_fMRI_out = fr'E:\PycharmProjects_E\SchemeRep\fMRI_in_BOLD\{sn}\{name}'
                dir_fMRI_out = Path(dir_fMRI_out)
                dir_fMRI_out.mkdir(parents=True, exist_ok=True)
                fp_fMRI_out = dir_fMRI_out / f'BOLD_run{run}.nii.gz'
                if os.path.exists(fp_fMRI_out):
                    continue

                fMRI_exists = fp_fMRI.exists()
                if fMRI_exists:
                    shutil.copyfile(fp_fMRI, fp_fMRI_out)
                continue



                fp_out = fr'cache/confounds/{sn}_{sess}_{run}_confounds.tsv'
                if os.path.exists(fp_out):
                    continue

                fp_confound = (sess_dir / 'func' /
                               f'sub-{sn}_ses-{sess}_task-{name}_run-0{run}_desc-confounds_timeseries.tsv')
                # print(f'{fp_confound=}')
                exists = fp_confound.exists()
                if exists:
                    fp_out = fr'cache/confounds/{sn}_{sess}_{run}_confounds.tsv'
                    shutil.copyfile(fp_confound, fp_out)
                else:
                    print(f'No confound file: {fp_confound}')




            fp_out = fr'cache/confounds/{sn}_resting_confounds.tsv'
            if os.path.exists(fp_out):
                did_rs = True
                continue

            fp_rs_confound = (sess_dir / 'func' /
                               f'sub-{sn}_ses-{sess}_task-resting_run-1_desc-confounds_timeseries.tsv')
            exists = fp_rs_confound.exists()
            if exists:
                did_rs = True
                shutil.copyfile(fp_rs_confound, fp_out)
        if not did_rs:
            print(f'No resting state csv for {sn}')
