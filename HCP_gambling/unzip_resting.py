import zipfile
import os

from os import fspath
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile
from tqdm.auto import tqdm  # could use from tqdm.gui import tqdm
from tqdm.utils import CallbackIOWrapper
import time

def get_sns():
    fns = os.listdir(r'F:\HCP_RS')
    fns = [fn for fn in fns if 'REST1_' in fn and fn[-4:] == '.zip']
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    return sns

def extractall(fzip, dest, desc="Extracting", override=False):
    """zipfile.Zipfile(fzip).extractall(dest) with progress"""
    dest = Path(dest).expanduser()
    with ZipFile(fzip) as zipf, tqdm(
        desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
        total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
    ) as pbar:
        for i in zipf.infolist():
            fn = os.path.split(i.filename)[-1]
            if '.nii' in fn or '.gii' in fn:
                if fn not in ['rfMRI_REST1_RL.nii.gz',
                              'rfMRI_REST1_LR.nii.gz',
                              'brainmask_fs.2.nii.gz']:
                    continue
            if os.path.isfile(fspath(dest / i.filename)) and not override:
                continue
            if fn == 'Movement_AbsoluteRMS_mean.txt': continue
            Path(dest / i.filename).parent.mkdir(parents=True, exist_ok=True)
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, fspath(dest))
            else:
                with zipf.open(i) as fi, open(fspath(dest / i.filename), "wb") as fo:
                    copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)

def unzip_all_resting(override=False, slow_check=True):
    sns = get_sns()

    # sns = {'150423', '171734', '119833', '127933', '128127',
    #            '127327', '105216', '105014', '203418', '203923',
    #            '204016', '201515', '201717', '201818', '202113'}

    # sns = ['144226']
    # from nilearn import image
    # fp = r'G:\HCP_gambling\144226\MNINonLinear\Results\tfMRI_GAMBLING_LR\tfMRI_GAMBLING_LR.nii.gz'
    # img = image.load_img(fp)
    # print(img.shape)
    # quit()
    # print(f'{len(sns)=}')
    # quit()
    # print(f'{len(sns)=}')
    # quit()
    # sns = ['100206']
    # sns = sns[::8]
    for sn in sns:
        # if sn == '106824' and not override:
        #     # errors
        #     continue
        fp_zip = rf'F:\HCP_RS\{sn}_3T_rfMRI_REST1_preproc.zip'
        missing_checks = [r'rfMRI_REST1_RL\rfMRI_REST1_RL.nii.gz',
                          r'rfMRI_REST1_LR\rfMRI_REST1_LR.nii.gz',
                          r'rfMRI_REST1_RL\brainmask_fs.2.nii.gz',
                          r'rfMRI_REST1_LR\brainmask_fs.2.nii.gz',
                          r'rfMRI_REST1_LR\Movement_Regressors.txt']
        if slow_check:
            do = False
            for check in missing_checks:
                fp_test_unzip = rf'F:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\{check}'
                if not os.path.exists(fp_test_unzip):
                    do = True
                    break
                if os.path.getsize(fp_test_unzip) < 1000:
                    do = True
                    break
                if 'LR.nii.gz' in check:
                    if os.path.getsize(fp_test_unzip) < 150 * 1e6:
                        do = True
                        break
            if not do and not override:
                print(f'Pass: {sn}')
                continue
        else:
            check = missing_checks[0]
            fp_test_unzip = rf'F:\HCP_RS_unzipped\{sn}\MNINonLinear\Results\{check}'
            if os.path.exists(fp_test_unzip) and not override:
                print(f'Pass: {fp_test_unzip=}')
                continue
                # print(f'Pass: {fp_test_unzip=}')
        # fp_test_unzip = rf'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
        # fp_test_unzip = rf'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\brainmask_fs.2.nii.gz'


        print(f'Unzipping: {sn}')
        dest_unzip = rf'G:\HCP_RS_unzipped'
        Path(dest_unzip).mkdir(exist_ok=True, parents=True)

        try:
            extractall(fp_zip, dest_unzip, override=True)
        except FileNotFoundError as e:
            print(f'File not found: {sn}, {e=}')
            time.sleep(5)
        except zipfile.BadZipfile:
            print(f'Bad zip file: {sn}')
        except OSError:
            print(f'OS error: {sn}')
            continue

        # with open(fp_zip, 'rb') as f:
        #     with zipfile.ZipFile(f) as z:
        #         z.extractall(fp_unzip)

if __name__ == '__main__':
    unzip_all_resting()