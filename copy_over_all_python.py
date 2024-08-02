import shutil
import glob
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    dir_out = r'X:\Studies\SchemeRep\PaulBogdan\Python'
    check_dirs = ['networks', 'finalizing', 'connRSA', 'EEG_fMRI', 'old',
                  ]
    fps_outer = glob.glob(rf'*.py')
    fps_all = set(fps_outer)
    for check_dir in check_dirs:
        fps = glob.glob(rf'{check_dir}/**/**/*.py', recursive=True)
        fps = set(fps)
        fps_all = fps_all.union(fps)

    for fp in tqdm(fps_all, desc='Copying over .py files'):
        fp_out = f'{dir_out}/{fp}'
        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fp, fp_out)