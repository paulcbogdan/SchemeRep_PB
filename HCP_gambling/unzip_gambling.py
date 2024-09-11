import zipfile
import os

def get_sns():
    fns = os.listdir(r'G:\HCP_G')
    fns = [fn for fn in fns if 'GAMBLING' in fn]
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    return sns

def unzip_all_gambling():
    sns = get_sns()
    for sn in sns:
        if sn == '106824':
            # errors
            continue
        fp_zip = rf'G:\HCP_G\{sn}_3T_tfMRI_GAMBLING_preproc.zip'
        fp_test_unzip = rf'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
        if os.path.exists(fp_test_unzip):
            continue
        print(f'Unzipping: {sn}')
        fp_unzip = rf'G:\HCP_gambling'
        with open(fp_zip, 'rb') as f:
            with zipfile.ZipFile(f) as z:
                z.extractall(fp_unzip)

if __name__ == '__main__':
    unzip_all_gambling()