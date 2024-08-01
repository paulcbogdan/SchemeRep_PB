import os
import shutil

dir_in = fr'H:\PycharmProjects_H\SchemeRep\fMRI_in'
sns = os.listdir(dir_in)
for sn in sns:
    if '.DS' in sn: continue
    print(sn)
    dir_sn_rest = fr'{dir_in}\{sn}\resting'
    if os.path.exists(dir_sn_rest):
        print(dir_sn_rest)
        shutil.rmtree(dir_sn_rest)
