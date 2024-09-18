import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

if __name__ == "__main__":
    fp_ns = r'C:\PycharmProjects\SchemeRep\HCP_gambling\HCP_individual_difs.csv'
    df_ns = pd.read_csv(fp_ns)

    cols = df_ns.columns
    cols = [col for col in cols if 'FS_' not in col]
    cols = [col for col in cols if 'NEORAW' not in col]
    cols = [col for col in cols if '_Compl' not in col]
    ns_cols = cols

    reg_global = True
    no_compcor = True
    anat_ver = 3
    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    fn_out = fr'fMRI_HCP_results_{anat_ver}{glob_str}{cc_str}.csv'
    fp_fMRI = fr'C:\PycharmProjects\SchemeRep\{fn_out}'
    df = pd.read_csv(fp_fMRI)
    df['Subject'] = df['sns']

    overlapping_sns = set(df['Subject']).intersection(set(df_ns['Subject']))
    print(f'{len(overlapping_sns)=}')

    df = df.merge(df_ns, on='Subject', how='inner')
    # ns_cols = ['DDisc_AUC_200', 'DDisc_AUC_40K']
    # print(df)
    # quit()
    plt.hist(df['rs_dd'])
    plt.show()

    print(f'{len(ns_cols)=}')
    fMRI_cols = ['rs_vendor', 'itr', 'rs_dd', 'rs_vv', 'rs_dv_ant', 'rs_dv_pos']
    for fMRI_col in fMRI_cols:
        for ns_col in ns_cols:
            df_ = df.dropna(subset=[fMRI_col, ns_col])
            N = len(df_)
            r, p = stats.spearmanr(df_[fMRI_col], df_[ns_col])
            if p < .005:
                print(f'{fMRI_col} x {ns_col} (N = {N}): {r=:.2f}, {p=:.4f}')
            # df[f'{fMRI_col}_{ns_col}'] = df[fMRI_col] - df[ns_col]