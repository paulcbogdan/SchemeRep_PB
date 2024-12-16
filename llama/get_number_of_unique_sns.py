from org_sns import get_sns
from organize_bhv import get_trial_info

if __name__ == '__main__':
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    dif_layouts = set()
    for sn in sns:
        df_sn = get_trial_info(sn)
        objs = df_sn['obj'].to_list()
        scns = df_sn['scene'].to_list()
        obj_scns = [(obj, scn) for obj, scn in zip(objs, scns)]
        obj_scns.sort()
        obj_scns = tuple(obj_scns)
        if obj_scns not in dif_layouts:
            print(f'Unique: {sn}')
            dif_layouts.add(obj_scns)
        else:
            print(f'\tDuplicate: {sn}')
