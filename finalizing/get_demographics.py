from old_Apr6.corr_RSA_x_vendor import get_plain_df_sn
import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')
import pandas as pd

def do_demo():
    df, _ = get_plain_df_sn()
    print(len(df['sn'].unique()))
    print(list(df['sn'].unique()))

    fp_in = r'behavFiles/SchemeRep_age_sex.csv'
    df_demo = pd.read_csv(fp_in)

    sns = set(df['sn'])


    df_demo['sn'] = df_demo['sn'].apply(str)

    demo_sns = set(df_demo['sn'])
    for sn in sns:
        if sn not in demo_sns:
            print(f'Missing {sn=}')

    df_demo['age_group'] = df_demo['sn'].apply(lambda x: x[0])
    df_demo['is_female'] = df_demo['gender'] == 'Female'
    df_demo = df_demo[df_demo['sn'].isin(sns)]
    M_age = df_demo.groupby('age_group')['age'].mean()
    print(M_age)
    SD_age = df_demo.groupby('age_group')['age'].std()
    print(SD_age)

    M_gender = df_demo.groupby('age_group')['is_female'].mean()
    print(M_gender)



if __name__ == '__main__':
    do_demo()
    quit()



    df, _ = get_plain_df_sn()
    print(df.columns)

    for inc in range(1, 4):
        df_inc = df[df['inc'] == inc]
        key = 'inc_rt'
        # key = 'per_inc'
        df_sn_M = df_inc.groupby('sn').mean(numeric_only=True)[key].mean()
        df_sn_SD = df_inc.groupby('sn').mean(numeric_only=True)[key].std()
        print(f'{inc=}: {df_sn_M=:.3f} | {df_sn_SD=:.3f}')
    quit()



