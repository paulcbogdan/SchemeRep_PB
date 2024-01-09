import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def calculate_partial_eta_squared(F, df1, df2):
    # reference: https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
    eta_squared = F*df1/(F*df1+df2)
    return eta_squared

def get_stars(p):
    if p < .001:
        stars = '***'
    elif p < .01:
        stars = '**'
    elif p < .05:
        stars = '*'
    elif p < .1:
        stars = '†'
    else:
        stars = 'NS'
    return stars

def get_t_F_p_eta_d(vals):
    M = vals.mean()
    SD = vals.std()
    N = len(vals)
    SE = SD / np.sqrt(N)
    t = M / SE
    F = t ** 2
    p = stats.t.sf(np.abs(t), N - 1) * 2
    eta_squared = calculate_partial_eta_squared(F, 1, N - 1)
    d = M / SD
    return t, F, p, eta_squared, d

def get_t_F_p_eta_d2(vals0, vals1):
    # two sample t-test with unequal variances
    M0 = vals0.mean()
    M1 = vals1.mean()
    SD0 = vals0.std()
    SD1 = vals1.std()
    N0 = len(vals0)
    N1 = len(vals1)
    SE = np.sqrt(SD0**2/N0 + SD1**2/N1)
    t = (M0 - M1) / SE
    F = t ** 2
    p = stats.t.sf(np.abs(t), N0 + N1 - 2) * 2
    eta_squared = calculate_partial_eta_squared(F, 1, N0 + N1 - 2)
    d = (M0 - M1) / np.sqrt((SD0**2 + SD1**2) / 2)
    return t, F, p, eta_squared, d

def setup_plots(plot_pair):
    if plot_pair:
        fig, axs = plt.subplots(1, 2, figsize=(8.75, 5),
                                gridspec_kw={'width_ratios': [2, 1],
                                             'left': .1, 'bottom': .2,
                                             'right': .98})
        plt.sca(axs[0])
    else:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5),
                                gridspec_kw={'width_ratios': [1],
                                             'left': .2, 'bottom': .18,
                                             'right': .95, 'top': .9})
        plt.sca(axs)
    return axs

def add_interaction_lines(y_high, y_low):

    lower_signif_line = (y_high - y_low) * .93 + y_low
    print(f'{lower_signif_line=}')
    upper_signif_line = (y_high - y_low) * .99 + y_low
    print(f'{upper_signif_line=}')
    plt.plot([0, 1], [lower_signif_line, lower_signif_line], color='k')
    plt.plot([0.5, 0.5], [lower_signif_line, upper_signif_line], color='k')
    plt.plot([2, 3], [lower_signif_line, lower_signif_line], color='k')
    plt.plot([2.5, 2.5], [lower_signif_line, upper_signif_line], color='k')
    plt.plot([0.5, 2.5], [upper_signif_line, upper_signif_line], color='k')
    return upper_signif_line, lower_signif_line

def run_ANOVA_ttest(df_scores, X, y_high, y_low, upper_signif_line):
    # This runs performs all of the ANOVA and t-test comparisons.
    #   Along with just printing output, the function adds asterisks to the bar
    #   graph to indicate significance.

    # --------- Interaction ---------
    interaction = df_scores[X[0]] - df_scores[X[1]] - \
                  df_scores[X[2]] + df_scores[X[3]]
    # _, F_intr, p_itr, eta_itr, _ = get_t_F_p_eta_d(interaction)
    _, F_intr, p_itr, eta_itr, _ = get_t_F_p_eta_d2(df_scores[X[0]] - df_scores[X[1]],
                                                    df_scores[X[2]] - df_scores[X[3]])

    print(f'Interaction effect: '
          f'{F_intr=:.3f}, {p_itr=:.4f}, {eta_itr=:.3f}')
    stars_itr = get_stars(p_itr)  # Add asterisks to bar graph
    fontsize = 20 if stars_itr in ['NS', '†'] else 24
    plt.text(1.5, (y_high - y_low) * .89 + y_low, stars_itr,
             fontsize=fontsize, ha='center')

    # --------- Test main effects ---------
    # Direction
    main_effect_of_direction = df_scores[X[0]] + df_scores[X[1]] - \
                               df_scores[X[2]] - df_scores[X[3]]
    _, F_direction, p_direction, eta_direction, _ = \
        get_t_F_p_eta_d(main_effect_of_direction)

    print(f'Main Direction effect: '
          f'{F_direction=:.3f}, {p_direction=:.4f}, {eta_direction=:.3f}')
    # Valence
    main_effect_val = df_scores[X[0]] - df_scores[X[1]] + \
                      df_scores[X[2]] - df_scores[X[3]]
    _, F_val, p_val, eta_val, _ = get_t_F_p_eta_d(main_effect_val)
    print(f'Main Valence effect: '
          f'{F_val=:.3f}, {p_val=:.3f}, {eta_val=:.3f}')

    # --------- post hoc t-tests ---------
    before_posthoc = df_scores[X[0]] - df_scores[X[1]]
    t_B, _, p_B, _, d_B = get_t_F_p_eta_d(before_posthoc)
    stars_B = get_stars(p_B)
    fontsize = 20 if stars_B in ['NS', '†'] else 24
    plt.text(0.5, (y_high - y_low) * .83 + y_low, stars_B,
             fontsize=fontsize, ha='center')
    print('Post hoc t-tests:')
    print(f'\tBefore: Emo vs Neu effect: '
          f'{t_B=:.3f}, p = {p_B:.3f}, {d_B=:.2f}')

    after_posthoc = df_scores[X[2]] - df_scores[X[3]]
    t_A, _, p_A, _, d_A = get_t_F_p_eta_d(after_posthoc)
    stars_A = get_stars(p_A)
    fontsize = 20 if stars_A in ['NS', '†'] else 24
    plt.text(2.5, (y_high - y_low) * .83 + y_low, stars_A,
             fontsize=fontsize, ha='center')
    print(f'\tAfter: Emo vs Neu effect: '
          f'{t_A=:.3f}, p = {p_A:.3f}, {d_A=:.2f}')

    neg_posthoc = df_scores[X[0]] - df_scores[X[2]]
    # t_neg, _, t_neg, _, d_neg = get_t_F_p_eta_d(neg_posthoc)
    t_neg, _, t_neg, _, d_neg = get_t_F_p_eta_d2(df_scores[X[0]],
                                                 df_scores[X[2]])

    print(f'\tEmo: Before vs. After effect {t_neg=:.3f}, p = {t_neg:.3f}')
    if t_neg < .05:
        stars_Emo = get_stars(t_neg)
        Emo_line = upper_signif_line + (y_high - y_low) * .14
        plt.plot([0, 2], [Emo_line, Emo_line], color='k')
        fontsize = 20 if 'NS' in stars_Emo else 24
        plt.text(1, - (y_high - y_low) * .005 + Emo_line, stars_Emo,
                 fontsize=fontsize, ha='center')
        highest_line = Emo_line + .03
    else:
        highest_line = upper_signif_line * 1.1

    neu_posthoc = df_scores[X[1]] - df_scores[X[3]]
    # t_neu, _, p_neu, _, d_neu = get_t_F_p_eta_d(neu_posthoc)
    t_neu, _, p_neu, _, d_neu = get_t_F_p_eta_d2(df_scores[X[1]],
                                                 df_scores[X[3]])
    print(f'\tNeu: Before vs. After effect {t_neu=:.3f}, p = {p_neu:.3f}')
    if p_neu < .05:
        stars_Neu = get_stars(p_neu)
        Neu_line = upper_signif_line + (y_high - y_low) * .1#0825
        plt.plot([1, 3], [Neu_line, Neu_line], color='k')
        fontsize = 20 if 'NS' in stars_Neu else 24
        plt.text(2, -(y_high - y_low) * .10 + Neu_line, stars_Neu,
                 fontsize=fontsize, ha='center')
        highest_line = max(highest_line, Neu_line * 1.06)
    return stars_itr, highest_line

def plot_right_side(df_scores, X, y_low, stars_itr):
    M2 = np.array([(df_scores[X[1]] + df_scores[X[2]]).mean() / 2,
                   (df_scores[X[0]] + df_scores[X[3]]).mean() / 2])
    SD2 = np.array([(df_scores[X[1]] + df_scores[X[2]]).std() / 2,
                    (df_scores[X[0]] + df_scores[X[3]]).std() / 2])
    SE2 = SD2 / np.sqrt(df_scores[X[1]].shape[0])
    upper_signif_line2 = (max(M2 + SE2) - y_low) * 1.1 + y_low
    upper_stars2 = (max(M2 + SE2) - y_low) * 1.09 + y_low
    plt.bar(['Negative\n-to-\nNeutral', 'Neutral\n-to-\nNegative'], M2 - y_low,
            yerr=SE2, color=['red', 'dodgerblue'], capsize=5,
            bottom=y_low)
    plt.plot([0., 1.], [upper_signif_line2, upper_signif_line2], color='k')
    fontsize = 20 if 'NS' in stars_itr else 24
    plt.text(0.5, upper_stars2, stars_itr, fontsize=fontsize, ha='center')
    plt.gca().spines[['right', 'top']].set_visible(False)

def print_mean_scores(df_sdt, X, key):
    print('Mean scores:')
    for x in X:
        y = df_sdt[x].mean()
        y_se = df_sdt[x].std() / np.sqrt(df_sdt[x].shape[0])
        print(f'\t {key} | {x}, {y:.1%} [SD = {y_se:.1%}]')

def run_tests_plot_bars(df_scores, key_to_label, replication=True):
    ylim_high = {'pFA': .52,
                 'corr_acc': .24,
                 'd': .7,
                 'c': 1.15}
    key_to_fn = {'d': 'd_sensitivity',
                 'c': 'c_criterion',
                 'pFA': 'prop_FA',
                 'corr_acc': 'corr_acc'}
    for key in list(key_to_label):
        str_rep = 'Repl_' if replication else 'Disc_'
        out_dir = r'figs_out'
        fn_out = fr'{out_dir}/{str_rep}{key_to_fn[key]}.png'
        if key in ['pFA', 'c']:
            plot_pair = False # controls whether one or two bar graphs plotted
                              # The two-bar graph figures are in the SuppMat
        else:
            plot_pair = True

        print('-'*80)
        X = ['emo_neu_B_{}', 'neu_emo_B_{}', 'emo_neu_A_{}', 'neu_emo_A_{}']
        X = [x.format(key) for x in X]
        print_mean_scores(df_scores, X, key)

        M = df_scores[X].mean()
        SE = df_scores[X].std() / np.sqrt(df_scores[X].shape[0])
        axs = setup_plots(plot_pair)
        X_labels = ['Neg Cue ', 'Neu Cue ', 'Neg Cue', 'Neu Cue']
        colors = ['dodgerblue', 'red', 'red', 'dodgerblue']
        plt.bar(X_labels, M, yerr=SE, color=colors, capsize=5)
        plt.ylabel(key_to_label[key], fontsize=17.5)
        plt.yticks(fontsize=15)

        y_low = 0
        y_high = ylim_high[key]


        upper_signif_line, lower_signif_line = \
            add_interaction_lines(y_high, y_low)

        stars_itr, highest_line = run_ANOVA_ttest(df_scores, X, y_high, y_low,
                                                  upper_signif_line)

        plt.gca().spines[['right', 'top']].set_visible(False)

        # Center x-labeled
        plt.text(0.5, y_low - (y_high - y_low)*.225, 'Target Before',
                 fontsize=17.5, ha='center')

        plt.text(2.5, y_low - (y_high - y_low)*.225, 'Target After',
                 fontsize=17.5, ha='center')

        line = plt.Line2D([0.5, 0.5], [-0.003, -.225],
                          transform=plt.gca().transAxes,
                          color='black',
                          dash_capstyle='butt')
        line.set_clip_on(False)
        plt.gca().add_line(line)

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)

        title = 'Replication\n' if replication else 'Discovery\n'
        if not plot_pair:
            plt.title(title, y=.95, fontsize=22)
            plt.savefig(fn_out, dpi=600)
            plt.show()
            continue

        plt.sca(axs[1])
        plot_right_side(df_scores, X, y_low, stars_itr)
        plt.suptitle(title, y=.98, fontsize=24)
        plt.ylim(bottom=y_low, top=highest_line)
        plt.savefig(fn_out, dpi=600)
        plt.show()

def run_analysis():
    font = {'size': 16.5}
    matplotlib.rc('font', **font)

    df_scores = get_df_scores()
    key_to_label = {'corr_acc': 'Corrected accuracy',
                    'pFA': 'False alarm proportion',
                    'd': 'd\' (sensitivity)',
                    'c': 'Criterion (c)'}
    run_tests_plot_bars(df_scores, key_to_label)


if __name__ == '__main__':
    run_analysis()
