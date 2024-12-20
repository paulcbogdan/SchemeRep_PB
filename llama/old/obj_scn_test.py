import pandas as pd

from org_sns import get_sns
from organize_bhv import get_trial_info

if __name__ == '__main__':
    # sns = get_sns('all')['healthy']
    # bad_sns = ['116', '125', '133', '213', '215', '231']
    # sns = [sn for sn in sns if sn not in bad_sns]
    #
    # for sn in sns:
    df_sn = get_trial_info('102')
    objs = df_sn['obj'].to_list()
    scns = df_sn['scene'].to_list()
    # print(objs)
    print(scns)

    # TODO: make .csv with obj + prior grammar
    # TODO: make .csv with scene + prior grammar

    df_obj = pd.DataFrame({f'obj': objs})
    df_obj.to_csv(r'llama/obj_no_grammar.csv', index=False)
    df_scn = pd.DataFrame({f'scene': scns})
    df_scn.to_csv(r'llama/scn_no_grammar.csv', index=False)
