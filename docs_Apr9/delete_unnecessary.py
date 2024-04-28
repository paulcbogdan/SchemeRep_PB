from pathlib import Path
import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')
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
    drop_dirs = ['all_BLruns_sorted',
                 'all_CONruns_sorted',
                 'all_VISruns_sorted',
                 'all_ENCruns_sorted',
                 'BL_rerun3',
                 'CON_rerun3',
                 'VIS_rerun3',
                 'ENC_rerun3',
                 ]

    dir_in = Path('fMRI_in')
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
                # continue
                shutil.rmtree(sn_data_dir)
            # print(dir_only, ':', dir_only in drop_dirs)




