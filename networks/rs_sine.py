from dFC_control_FC import get_all_task_vendor
from factor_analysis import identify_extremely_low_variance_sn


def do_rs_sine(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor(anat=True, zscore=True)
    # df = df[df['task'] != 'RS']
    # df = df[df['task'] == 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    bad_sns.add('138')
    bad_sns.add('217') # almost always near zero for Horz
    bad_sns.add('110') # almost always nere zero for Vendor
    df = df[~df['sn'].isin(bad_sns)]
    df.dropna(subset=['da', 'dp', 'va', 'vp'], inplace=True)

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    df['vendor'] = df['dd_vv'] - df['dv_dv']



if __name__ == '__main__':
    do_rs_sine()