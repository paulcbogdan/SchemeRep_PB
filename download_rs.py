from tqdm import tqdm

from analyze_rs import analyze_vendor
from org_sns import get_sns

# def get_rs_vendor_df():


# def analyze_hemi_vendor():


if __name__ == '__main__':
    dir_in = r'Z:\Cabeza\SchemRep.01\Data\fMRIprep_by_subject_out'
    sns = get_sns()
    sns = sns[1] + sns[2]
    for sn in tqdm(sns, desc='Looping over fMRI'):
        has_match = False
        for i in range(1, 4):
            dir_sn = rf'{dir_in}/sub-{sn}/ses-{i}/func'
            d = Path(dir_sn)
            # fp = f'{dir_sn}/sub-{sn}_ses-{i}_task-resting_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
            fp = f'{dir_sn}/sub-{sn}_ses-{i}_task-resting_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

            match = os.path.isfile(fp)
            if match:
                fp_out = fr'E:/PycharmProjects_E/SchemeRep/fMRI_in/{sn}/resting/rs0.nii.gz'
                if os.path.isfile(fp_out):
                    print(f'Already exists: {sn}')
                    break
                Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
                print(f'Copying: {sn}')
                shutil.copyfile(fp, fp_out)
                break
        else:
            print(f'{sn}: no match')