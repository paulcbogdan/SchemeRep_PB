from org_sns import get_sns
from organize_bhv import get_trial_info
import numpy as np

if __name__ == '__main__':
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    sns = ['102']

    con_old_hit_rates = []
    vis_old_hit_rates = []
    vis_similar_hit_rates = []
    BL_resps = []
    for sn in sns:
        df = get_trial_info(sn, verbose=-1,
                            easy_override=True)
        df_vis_old = df[df['vis_type'] == 'old']
        vis_old_hit_rate = df_vis_old['vis_hit'].mean()
        vis_old_hit_rates.append(vis_old_hit_rate)
        df_vis_similar = df[df['vis_type'] == 'similar']
        vis_similar_hit_rate = df_vis_similar['vis_hit'].mean()
        vis_similar_hit_rates.append(vis_similar_hit_rate)
        con_old_hit_rate = df['con_hit'].mean()
        con_old_hit_rates.append(con_old_hit_rate)
        BL_resp = df['bl_resp'].mean()
        print(df['bl_resp'])
        quit()
        BL_resps.append(BL_resp)
        print(BL_resp)
        # print(df_vis_similar['vis_hit'].value_counts())
        # print(f'{len(df)=}')
        # print(df['vis_type'].value_counts())
        # print(df.columns)
    M_vis_old_hit_rate = np.mean(vis_old_hit_rates)
    SD_vis_old_hit_rate = np.std(vis_old_hit_rates)
    print(f'Visual old: M = {M_vis_old_hit_rate:.2f} [SD = {SD_vis_old_hit_rate:.2f}]')
    M_vis_similar = np.mean(vis_similar_hit_rates)
    SD_vis_similar = np.std(vis_similar_hit_rates)
    print(f'Visual similar: M = {M_vis_similar:.2f} [SD = {SD_vis_similar:.2f}]')
    M_con_old = np.mean(con_old_hit_rates)
    SD_con_old = np.std(con_old_hit_rates)
    print(f'Conceptual old: M = {M_con_old:.2f} [SD = {SD_con_old:.2f}]')
    M_BL_resp = np.mean(BL_resps)
    SD_BL_resp = np.std(BL_resps)
    print(f'BL_resp: M = {M_BL_resp:.2f} [SD = {SD_BL_resp:.2f}]')
