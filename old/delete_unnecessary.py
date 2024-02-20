from pathlib import Path
import os
import shutil

if __name__ == '__main__':
    drop_dirs = ['BL', 'BL_rerun',
                 'CON_rerun',
                 'dif_bl_obj',
                 'dif_bl_vis',
                 'dif_obj_vis',
                 'ENC',
                 'Enc_rerun',
                 'VIS_rerun']

    dir_in = Path('../fMRI_in')
    sns = dir_in.glob('*')
    for sn_dir in sns:
        sn_only = sn_dir.name
        # sn_dir = dir_in.joinpath(sn)
        # print(sn_dir)
        sn_data_dirs = sn_dir.glob('*')
        # print(list(sn_data_dirs))
        for sn_data_dir in sn_data_dirs:
            dir_only = sn_data_dir.name
            if dir_only in drop_dirs:
                print(f'Deleting {sn_only}/{dir_only}')
                shutil.rmtree(sn_data_dir)
            # print(dir_only, ':', dir_only in drop_dirs)




