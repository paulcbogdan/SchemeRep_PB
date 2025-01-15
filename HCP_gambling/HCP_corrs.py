import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == "__main__":
    fp_ns = r'C:\PycharmProjects\SchemeRep\HCP_gambling\HCP_individual_difs.csv'
    df_ns = pd.read_csv(fp_ns)
    print(df_ns['Subject'])
    quit()

    cols = df_ns.columns
    cols = [col for col in cols if 'FS_' not in col]
    cols = [col for col in cols if 'NEORAW' not in col]
    cols = [col for col in cols if '_Compl' not in col]
    dd_cols = [col for col in cols if 'DDisc' in col]

    for age, df_age in df_ns.groupby('Age'):
        print(f'{age}: N = {len(df_age)}')

    for dd_col in dd_cols:
        df_ns[dd_col] = stats.zscore(df_ns[dd_col], nan_policy='omit')
        dd_col_str = f'{dd_col:<20} | '
        for age, df_age in df_ns.groupby('Age'):
            if age == '36+': continue
            M = df_age[dd_col].mean()
            SE = df_age[dd_col].std() / (df_age[dd_col].count() ** .5)
            dd_col_str += f'{M=:>6.2f} ({SE=:.2f}), '
        dd_col_str = dd_col_str[:-2]
        print(dd_col_str)
        # print(f'{dd_col} ({age}): {M=:.2f} ({SE=:.2f}), {N=}')

    # print(df_ns['Age'].value_counts())
    quit()
    ns_cols = cols

    reg_global = False
    no_compcor = False
    anat_ver = 3
    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    fn_out = fr'fMRI_HCP_results_{anat_ver}{glob_str}{cc_str}.csv'
    fp_fMRI = fr'C:\PycharmProjects\SchemeRep\{fn_out}'

    df = pd.read_csv(fp_fMRI)
    print(len(df))
    quit()
    df['Subject'] = df['sns']

    overlapping_sns = set(df['Subject']).intersection(set(df_ns['Subject']))
    print(f'{len(overlapping_sns)=}')

    df = df.merge(df_ns, on='Subject', how='inner')
    # ns_cols = ['DDisc_AUC_200', 'DDisc_AUC_40K']
    # print(df)
    # quit()

    # plt.scatter(df['rs_vendor'], df['Age'], alpha=.5)
    # plt.show()
    # quit()

    print(f'{len(ns_cols)=}')
    fMRI_cols = ['rs_vendor', 'itr', 'rs_dd', 'rs_vv', 'rs_dv_ant', 'rs_dv_pos',
                 'dd', 'vv', 'dv_ant', 'dv_pos', 'p_changes', 'PE_bhv_efs']
    for fMRI_col in fMRI_cols:
        plt.title(fMRI_col)
        plt.hist(df[fMRI_col])
        plt.show()
        for ns_col in ns_cols:
            df_ = df.dropna(subset=[fMRI_col, ns_col])
            N = len(df_)
            r, p = stats.spearmanr(df_[fMRI_col], df_[ns_col])
            if p < .01:
                print(f'{fMRI_col} x {ns_col} (N = {N}): {r=:.2f}, {p=:.4f}')
            # df[f'{fMRI_col}_{ns_col}'] = df[fMRI_col] - df[ns_col]