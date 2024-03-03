import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats as stats

from emotemporal_bar import get_stars
# from atlas_utils import get_atlas
from old.modularity import get_modules, get_partition_matrix
from old.plot_gen import plot_connectivity
from utils import pickle_wrap
from vendor_lmers import get_hemi_vendor_df, get_vendor_df, get_hemi_cross_vendor_df

# filter PerformanceWarning
import warnings
# import PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def plot_2x2_triangle(fp='obj7_fMRI', plot=True, hemi=True, scrub=False,
                      anat=True):
    df, vndr_cols = pickle_wrap(get_vendor_df, None, kwargs={'fp': fp,
                                                             'scrub': scrub,
                                                             'anat': anat,
                                                             'hemis': hemi},
                                easy_override=False, cache_dir='cache')
    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    df['age_str'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')

    sides_names = {'dd_vv': 'Ven-Dor\n(Horizontal)',
                   'dv_dv': 'Post-Ant\n(Vertical)',
                   'age_str': 'Age',
                   }
    sides = ['Ven-Dor\n(Horizontal)', 'Post-Ant\n(Vertical)']
    df = df.rename(columns=sides_names)

    df_names = df[list(sides_names.values())]
    sns.set(font_scale=1.15)
    g = sns.pairplot(df_names,
                     hue='Age',
                     kind="reg",
                     palette={'YA': 'green', 'OA': 'magenta'},
                     plot_kws={'scatter_kws': {'alpha': .05,
                                               }},  # 'color': 'orange'
                     diag_kws={'common_norm': False})
    df_ya = df[df['Age'] == 'YA'].dropna(subset=sides)
    df_oa = df[df['Age'] == 'OA'].dropna(subset=sides)
    for i, s0 in enumerate(sides):
        for j, s1 in enumerate(sides):
            if i == j:
                continue
            g.axes[i][j].set_xlim(-11, 11)
            g.axes[i][j].set_ylim(-11, 11)
            r_ya, p_ya = stats.pearsonr(df_ya[s0], df_ya[s1])
            r_oa, p_oa = stats.pearsonr(df_oa[s0], df_oa[s1])
            # g.axes[i][j].get_children()[3].set_alpha(.01)

            # if abs(r_oa - r_ya) < .1:
            #     for k in range(6):
            #         if k % 2 == 0: continue
            #         g.axes[i][j].get_children()[k].set_alpha(.001)
                    # g.axes[i][j].get_children()[k].set_color("black")
    for lh in g._legend.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [50]
    # plt.savefig('2x2_vendor_postant_triangle.png', dpi=300)
    # plt.show()



def lmer_triangle(fp='obj7_fMRI', plot=True, hemi=True, scrub=False, anat=True):
    df, vndr_cols = pickle_wrap(get_vendor_df, None, kwargs={'fp': fp,
                                                             'scrub': scrub,
                                                             'anat': anat,
                                                             'hemis': hemi},
                                easy_override=True, cache_dir='cache')

    # sides = ['dd', 'vv', 'dv_ant', 'dv_pos']
    sides = ['dd', 'vv', 'dv_pos', 'dv_ant']
    if hemi: sides += ['DP_hemi', 'VP_hemi', 'DA_hemi', 'VA_hemi']

    for dv in sides:
        df[dv] = stats.zscore(df[dv], nan_policy='omit')
    # df.loc[df[dv].abs() > 10] = np.nan
    df['hit_hit'] = stats.zscore(df['hit_hit'], nan_policy='omit')
    df['age_str'] = df['age'].apply(lambda x: 'YA' if x == 1 else 'OA')
    df['age'] = stats.zscore(df['age'], nan_policy='omit')
    df['brain_M'] = stats.zscore(df['brain_M'], nan_policy='omit')
    df['inc'] = stats.zscore(df['inc'], nan_policy='omit')

    df.reset_index(inplace=True, drop=True)
    # print(df[sides])
    # quit()

    corr = df[sides].corr()
    print(corr)
    # quit()
    df[sides].dropna(inplace=True)
    if plot:
        sides_names = {'dd': 'Dorsal',
                       'vv': 'Ventral',
                       'dv_ant': '(Anterior)\nDorsal x Ventral',
                       'dv_pos': '(Posterior)\nDorsal x Ventral',
                       'age_str': 'Age',
                       'inc_str': 'inc_str',
                       'DP_hemi': 'DorPos Hemi',
                       'DA_hemi': 'DorAnt Hemi',
                       'VP_hemi': 'VenPos Hemi',
                       'VA_hemi': 'VenAnt Hemi'}
        df_names = df.rename(columns=sides_names)
        df_names = df_names[map(sides_names.get, sides +
                                ['age_str', 'inc_str'])]
        sns.set(font_scale=1.15)
        g = sns.pairplot(df_names,
                         hue='Age',
                         kind="reg",
                         # palette = 'orange',
                         # palette={'YA': 'dodgerblue', 'OA': 'red'},
                         palette={'YA': 'green', 'OA': 'magenta'},
                         plot_kws={'scatter_kws': {'alpha': .05,
                                                   }}, # 'color': 'orange'
                                   # 'line_kws': {'color': 'orange'}},
                         diag_kws={'common_norm': False})
        df_ya = df[df['age_str'] == 'YA'].dropna(subset=sides)
        df_oa = df[df['age_str'] == 'OA'].dropna(subset=sides)
        for i, s0 in enumerate(sides):
            for j, s1 in enumerate(sides):
                if i == j:
                    continue
                g.axes[i][j].set_xlim(-11, 11)
                g.axes[i][j].set_ylim(-11, 11)
                r_ya, p_ya = stats.pearsonr(df_ya[s0], df_ya[s1])
                r_oa, p_oa = stats.pearsonr(df_oa[s0], df_oa[s1])
                if abs(r_oa - r_ya) < .1:
                    for k in range(6):
                        g.axes[i][j].get_children()[k].set_color("black")
                    # g.axes[i][j].get_lines()[0].set_color("black")
                    # g.axes[i][j].get_lines()[1].set_color("black")
                    # g.axes[i][j].get_children()[0].set_color('black')
                    # g.axes[i][j].get_children()[3].set_color('black')
        for lh in g._legend.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [50]
        plt.show()

    ar_main = []
    ar_itr = []
    for i, v in enumerate(sides):
        row_main = []
        row_itr = []
        for j, w in enumerate(sides):
            if i == j:
                row_main.append('-')
                row_itr.append('-')
                continue
            # print(f'{v} x {w}')
            beta_main, p_main, beta_itr, p_itr = lmer4matrix(df, v, w,
                                                             hemi=hemi)
            stars_main = get_stars(p_main, pad=True)
            stars_itr = get_stars(p_itr, pad=True)
            row_main.append(f'{beta_main:.2f} {stars_main}')
            row_itr.append(f'{beta_itr:.2f} {stars_itr}')
            print(f'{v} x {w}: {beta_main=:.2f}, {p_main=:.3f}| '
                  f'{beta_itr=:.2f}, {p_itr=:.3f}')
        ar_main.append(row_main)
        ar_itr.append(row_itr)
    df_main = pd.DataFrame(ar_main, columns=sides, index=sides)
    print(df_main)
    print('-'*50)
    df_itr = pd.DataFrame(ar_itr, columns=sides, index=sides)
    print(df_itr)


def lmer4matrix(df, dv, iv, hemi=False, random_slops=True):
    pd.set_option('display.precision', 4)
    np.set_printoptions(precision=4)
    formula_gen = '{dv} ~ 1 + vv + dd + dv_ant + dv_pos + ' \
                  'DP_hemi + DA_hemi + VP_hemi + VA_hemi + ' \
                  '{itr} + inc + brain_M + (1 + {itr} | sn)'
    if not hemi:
        formula_gen = \
            formula_gen.replace(' + DP_hemi + DA_hemi + VP_hemi + VA_hemi', '')
    formula_gen = '{dv} ~ {itr} + inc + brain_M + (1 + {itr} | sn)'
    formula_gen = formula_gen.replace(f'{dv} + ', '')
    itr = f'{iv}*hit_hit'
    formula = formula_gen.format(dv=dv, itr=itr)

    from pymer4 import Lmer
    model = Lmer(formula, data=df)
    try:
        model.fit(REML=True, verbose=False, summary=False)
    except Exception as e:
        print(e)
        print('BAD!!')
        print(f'{formula=}')
        return np.nan, np.nan, np.nan, np.nan
    print(model.summary())
    summary = model.coefs
    print('-----------')
    beta_main = summary.loc[iv, 'Estimate']
    p_main = summary.loc[iv, 'P-val']
    # beta_itr = 0
    # p_itr = 1
    beta_itr = summary.loc[itr.replace('*', ':'), 'Estimate']
    p_itr = summary.loc[itr.replace('*', ':'), 'P-val']
    return beta_main, p_main, beta_itr, p_itr


def plot_massive_hemi_corr_matrix(fp='obj7_fMRI', anat=True, scrub=False):
    df, cols = pickle_wrap(get_hemi_vendor_df, None, kwargs={'fp': fp,
                                                             'anat': anat,
                                                             'scrub': scrub}, easy_override=False, cache_dir='cache')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 2)
    pd.options.display.float_format = '{:.2f}'.format

    cols_order = []
    cols_order_dd = ['Ldp_Lda', 'Rdp_Rda', # top = main direction
                     'Ldp_Rdp', 'Lda_Rda', # middle = hemi-bounce
                     'Ldp_Rda', 'Lda_Rdp'] # bottom = cross
    cols_order_vv = [col.replace('d', 'v') for col in cols_order_dd]
    cols_order_p  = ['Ldp_Lvp', 'Rdp_Rvp',
                     'Ldp_Rdp', 'Lvp_Rvp',
                     'Ldp_Rvp', 'Lvp_Rdp']
    cols_order_a  = [col.replace('p', 'a') for col in cols_order_p]

    cols_order_L = ['Ldp_Lda', 'Lvp_Lva',
                    'Ldp_Lvp', 'Lda_Lva',
                    'Ldp_Lva', 'Lda_Lvp']
    cols_order_R = [col.replace('L', 'R') for col in cols_order_L]

    cols_order += cols_order_dd + cols_order_vv + cols_order_p + cols_order_a
    # cols_order += cols_order_L + cols_order_R

    # cols_order = ['da_dp', 'va_vp', 'dp_vp', 'da_va',
    #               'Ldp_Rdp', 'Lvp_Rvp', 'Lda_Rda', 'Lva_Rva', ]
    #
    # cols_order = ['Lda_Ldp', 'Rda_Rdp',
    #               'Lva_Lvp', 'Rva_Rvp',
    #               'Ldp_Lvp', 'Rdp_Rvp',
    #               'Lda_Lva', 'Rda_Rva',
    #               'Ldp_Rdp', 'Lvp_Rvp', 'Lda_Rda', 'Lva_Rva', ]

    # cols_order += ['dd', 'vv']

    df.dropna(subset=cols_order, inplace=True)

    cols_within = ['Ldp_Ldp', 'Lda_Lda', 'Lvp_Lvp', 'Lva_Lva',
                   'Rdp_Rdp', 'Rda_Rda', 'Rvp_Rvp', 'Rva_Rva']
    # cols_order += cols_within

    for col in cols_order:
        M = df[col].mean()
        if col[1:3] != col[5:7]:
            continue

        height = '*' * int(abs(M) * 100)
        print(f'{col}: {M:+.2f} | {height}')

    tick_lows = np.arange(0, len(cols_order))
    ticks = tick_lows# + 0.5
    tick_labels = cols_order

    print(df[cols_order].corr())

    corr = np.array(df[cols_order].corr())
    corr[corr > .99] = np.nan

    median = np.nanmedian(corr)
    print(f'{median=}')

    corr[corr > 0] = 1
    corr[corr < 0] = 0
    partitions = get_modules(corr)

    # plt.imshow(corr)
    # plt.show()

    corr_v0 = get_partition_matrix(np.ones(corr.shape), partitions[0],
                                   w_zeros=True)

    for i, p in enumerate(partitions):
        p_named = [cols_order[j] for j in p]
        print(f'{i}: {p=} ({p_named})')
    # quit()

    corr_v1 = get_partition_matrix(np.ones(corr.shape), partitions[1],
                                   w_zeros=True)

    # corr = np.array(corr)
    plot_connectivity(corr_v0, ticks, tick_labels, tick_lows, title=fp, no_avg=True, vmin=-0.3, vmax=0.3)

    plot_connectivity(corr_v1, ticks, tick_labels, tick_lows, title=fp, no_avg=True, vmin=-0.3, vmax=0.3)


def plot_meta_corr_matrix(fp='con7_fMRI', hemis=True):
    if hemis:
        df, vndr_cols = pickle_wrap(get_vendor_df, None, kwargs={'fp': fp}, easy_override=False, cache_dir='cache')

        df_hemi, cols_hemi = pickle_wrap(get_hemi_cross_vendor_df, None, kwargs={'fp': fp}, easy_override=True,
                                         cache_dir='cache')
        df = df.merge(df_hemi, on=['sn', 'obj'])
        # cols = ['dd', 'vv', 'dv_ant', 'dv_pos'] + cols_hemi
        cols = ['dd', 'vv', 'dv_ant', 'dv_pos',
                'DP_hemi', 'DA_hemi', 'VP_hemi', 'VA_hemi']

    else:
        df, vndr_cols = pickle_wrap(get_vendor_df, None, kwargs={'fp': fp}, easy_override=False, cache_dir='cache')
        cols = ['dd', 'vv', 'dv_ant', 'dv_pos']
    for age, df_age in df.groupby('age'):
        print(age, ':', len(df_age['sn'].unique()))
        print(df_age['sn'].unique())
    quit()
    # TODO: for the DV_pos/ant x dd/vv, make sure that there are no overlapping
    #   edges
    corr = df[cols].corr()


    np.set_printoptions(edgeitems=10)
    np.set_printoptions(linewidth=200)
    ar_str = np.full((len(cols), len(cols)), '', dtype=object)
    for i, row in enumerate(corr.values):
        for j, r in enumerate(row):
            if r > .99999:
                ar_str[i][j] = ''
                continue
            z = np.arctanh(r)
            z_std = 1 / np.sqrt(len(df) - 3)
            z_low = z - 1.96 * z_std
            z_high = z + 1.96 * z_std
            r_low = np.tanh(z_low)
            r_high = np.tanh(z_high)
            # ar_str[i][j] = r
            ar_str[i][j] = f'{r:.2f}'# ({r_low:.2f}, {r_high:.2f})'
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df_main = pd.DataFrame(ar_str, columns=cols, index=cols)
    print(df_main)
    print('-')
    print(f'{fp=}')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 2)
    pd.options.display.float_format = '{:.2f}'.format
    plot_2x2_triangle()
    # lmer_triangle()
    # plot_massive_hemi_corr_matrix()
    # quit()
    # lmer_triangle()
    # plot_meta_corr_matrix()
