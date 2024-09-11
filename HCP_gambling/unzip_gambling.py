import zipfile
import os

from os import fspath
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile
from tqdm.auto import tqdm  # could use from tqdm.gui import tqdm
from tqdm.utils import CallbackIOWrapper

def get_sns():
    fns = os.listdir(r'G:\HCP_G')
    fns = [fn for fn in fns if 'GAMBLING' in fn]
    sns = {fn.split('_')[0] for fn in fns}
    sns = sorted(list(sns))
    return sns

def extractall(fzip, dest, desc="Extracting"):
    """zipfile.Zipfile(fzip).extractall(dest) with progress"""
    dest = Path(dest).expanduser()
    with ZipFile(fzip) as zipf, tqdm(
        desc=desc, unit="B", unit_scale=True, unit_divisor=1024,
        total=sum(getattr(i, "file_size", 0) for i in zipf.infolist()),
    ) as pbar:
        for i in zipf.infolist():
            fn = os.path.split(i.filename)[-1]
            if '.nii' in fn or '.gii' in fn:
                if fn not in ['tfMRI_GAMBLING_RL.nii.gz',
                              'tfMRI_GAMBLING_LR.nii.gz',
                              'brainmask_fs.2.nii.gz']:
                    continue
            # print(f'Extract: {fn}')
            if os.path.isfile(fspath(dest / i.filename)):
                # print('already did')
                continue
            # print(f'Extract: {fn}')
            if not getattr(i, "file_size", 0):  # directory
                zipf.extract(i, fspath(dest))
            else:
                with zipf.open(i) as fi, open(fspath(dest / i.filename), "wb") as fo:
                    copyfileobj(CallbackIOWrapper(pbar.update, fi), fo)

def unzip_all_gambling(override=True):
    sns = get_sns()
    if override:
        sns = ['106319', '106824', '111009', '111312']
    for sn in sns:
        if sn == '106824' and not override:
            # errors
            continue
        fp_zip = rf'G:\HCP_G\{sn}_3T_tfMRI_GAMBLING_preproc.zip'
        fp_test_unzip = rf'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\tfMRI_GAMBLING_RL.nii.gz'
        fp_test_unzip = rf'G:\HCP_gambling\{sn}\MNINonLinear\Results\tfMRI_GAMBLING_RL\brainmask_fs.2.nii.gz'
        if os.path.exists(fp_test_unzip) and not override:
            continue
        print(f'Unzipping: {sn}')
        fp_unzip = rf'G:\HCP_gambling'

        # try:
        extractall(fp_zip, fp_unzip)
        # except FileNotFoundError as e:
        #     print(f'File not found: {sn}, {e=}')
        # except ZipFile.BadZipFil
        #     print(f'Bad zip file: {sn}')

        # with open(fp_zip, 'rb') as f:
        #     with zipfile.ZipFile(f) as z:
        #         z.extractall(fp_unzip)

if __name__ == '__main__':
    unzip_all_gambling()