import numpy as np
from matplotlib import pyplot as plt
from statsmodels.formula import api as smf


def plot_different_IRAFs(df, ROI_cols):
    plt.hist(np.reshape(df[ROI_cols], -1), bins=400, density=True,
             range=(-1, 1), label='ROI', alpha=0.9, color='dodgerblue')
    plt.hist(df['BOLD_score'], bins=400, density=True, label='Region',
             range=(-1, 1), alpha=0.66, color='red')
    n, _, _ = plt.hist(df['conn_score'], bins=400, density=True, label='conn',
             range=(-1, 1), alpha=0.34, color='green')
    plt.plot([0, 0], [0, max(n)], 'k--', linewidth=0.75)
    plt.legend()
    plt.show()


def pie_charts(df, ROI_cols, title):
    df['ROI_M'] = df[ROI_cols].mean(axis=1)

    # df = df[cols_keep].dropna()
    mod = smf.ols(formula='conn_score ~ 1', data=df)
    res = mod.fit()
    conn_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{conn_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='conn_score ~ 1 + BOLD_score', data=df)
    res = mod.fit()
    conn_itr1 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress out BOLD {conn_itr1=:.4f} ({p=:.3f}, {z=:.3f})')




    mod = smf.ols(formula='conn_score ~ 1 + BOLD_score + '
                          + ' + '.join(ROI_cols),
                  data=df)
    res = mod.fit()

    conn_itr_r = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress out all: {conn_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    print('-')

    if p > 0.1:
        conn_itr_r = 0.

    # df['BOLD_score'] /= df['BOLD_score'].std()

    mod = smf.ols(formula='BOLD_score ~ 1', data=df)
    res = mod.fit()
    bold_itr0 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'{bold_itr0=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='BOLD_score ~ 1 + conn_score', data=df)
    res = mod.fit()
    bold_itr1 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress conn: {bold_itr1=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='BOLD_score ~ 1 + ROI_M', data=df)
    res = mod.fit()
    bold_itr2 = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress ROI_M: {bold_itr2=:.4f} ({p=:.3f}, {z=:.3f})')

    mod = smf.ols(formula='BOLD_score ~ 1 + conn_score + ' +
                          ' + '.join(ROI_cols),
                  data=df)
    res = mod.fit()
    bold_itr_r = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'\tRegress out all: {bold_itr_r=:.4f} ({p=:.3f}, {z=:.3f})')
    print('-')

    # print(res.summary())
    #

    # df['ROI_M'] /= df['ROI_M'].std()

    mod = smf.ols(formula='ROI_M ~ 1', data=df)
    res = mod.fit()
    ROI_M_itr0 = res.params['Intercept']
    print(f'{ROI_M_itr0=:.4f}')

    mod = smf.ols(formula='ROI_M ~ 1 + BOLD_score', data=df)
    res = mod.fit()
    ROI_M_itr1 = res.params['Intercept']
    print(f'\tRegress BOLD: {ROI_M_itr1=:.4f}')
    # print(res.summary())
    # quit()

    ctrl = '+ BOLD_score + conn_score'
    mod = smf.ols(formula=f'{ROI_cols[0]} ~ 1 {ctrl}', data=df)
    res = mod.fit()
    ROI_itr_total = res.params['Intercept']
    p = res.pvalues['Intercept']
    z = res.tvalues['Intercept']
    print(f'First ROI: {ROI_itr_total=:.4f} ({p=:.3f}, {z=:.3f})')
    ROI_itr_total_sign = np.sign(ROI_itr_total)
    ROI_itr_total = (ROI_itr_total ** 2) * ROI_itr_total_sign
    # print(res.summary())
    # quit()

    ROI_col_str = ROI_cols[0]
    for i in range(1, len(ROI_cols)):
        formula = f'{ROI_cols[i]} ~ 1 + {ROI_col_str} {ctrl}'
        mod = smf.ols(formula=formula, data=df)
        res = mod.fit()
        # print(res.summary())
        ROI_col_str += f' + {ROI_cols[i]}'
        ROI_itr = res.params['Intercept']

        ROI_itr_sign = np.sign(ROI_itr)
        ROI_itr_ = (ROI_itr ** 2) * ROI_itr_sign
        ROI_itr_total += ROI_itr_
        if True:
            p = res.pvalues['Intercept']
            z = res.tvalues['Intercept']
            ROI = ROI_cols[i]
            print(f'\t{ROI} | {ROI_itr:+.4f} = {np.sqrt(ROI_itr_total)=:.4f} '
                  f'({p=:.3f}, {z=:.3f})')

    print(f'{np.sqrt(ROI_itr_total)=:.4f}')
    # print(f'{ROI_itr_total=}')
    ROI_itr_r = np.sqrt(ROI_itr_total)
    print('-*-')
    print(f'{conn_itr_r=:.4f}')
    print(f'{bold_itr_r=:.4f}')
    print(f'{ROI_itr_r=:.4f}')
    plot_pie_chart([conn_itr_r, bold_itr_r, ROI_itr_r], title)


def plot_pie_chart(vals, title):
    plt.title(title, fontsize=24)
    cs_ = ['dodgerblue', 'red', 'green']
    sizes_ = vals#[conn_itr_r, bold_itr_r, ROI_itr_r]
    labels_ = ['Conn', 'Region', 'ROIs']
    cs, sizes, labels = [], [], []
    for c, s, l in zip(cs_, sizes_, labels_):
        if s > 0:
            cs.append(c)
            sizes.append(s)
            labels.append(l)

    wedge, text \
        = plt.gca().pie(sizes,
            labels=labels,
            colors=cs,
            textprops=dict(color='k', fontsize=24),
            labeldistance=1.1, wedgeprops={"alpha": 0.8,
                                           'edgecolor': 'w',
                                           'linewidth': 3.0},
            startangle=(225 if len(sizes) == 1 else -26),) #  +
    [autotext.set_color(c) for autotext, c in zip(text, cs)]
    plt.tight_layout()
    plt.show()
