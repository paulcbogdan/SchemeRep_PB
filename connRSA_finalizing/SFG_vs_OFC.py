from connRSA_finalizing.plot_RSM_x_RSM_bars import plot_FigureS1_bars_region
from scipy import stats

def test_SFG_vs_OFG(big_voxelwise=True):
    SFG_betas = plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=big_voxelwise,
                                          get_betas=True)
    OFG_betas = plot_FigureS1_bars_region('OrG', big_voxelwise=big_voxelwise,
                                          get_betas=True)

    conds = [(False, 'Local'), (False, 'Distributed'),
             (True, 'Local'), (True, 'Distributed')]

    SFG_itr = (SFG_betas[conds[0]] - SFG_betas[conds[1]] -
               SFG_betas[conds[2]] + SFG_betas[conds[3]])
    t_SFG_itr, p_SFG_itr = stats.ttest_1samp(SFG_itr, 0)
    SFG_per_ef = SFG_betas[conds[0]] - SFG_betas[conds[1]]
    SFG_per_t, SFG_per_p = stats.ttest_1samp(SFG_per_ef, 0)
    SFG_sem_ef = SFG_betas[conds[2]] - SFG_betas[conds[3]]
    SFG_sem_t, SFG_sem_p = stats.ttest_1samp(SFG_sem_ef, 0)

    OFG_itr = (OFG_betas[conds[0]] - OFG_betas[conds[1]] -
               OFG_betas[conds[2]] + OFG_betas[conds[3]])
    t_OFG_itr, p_OFG_itr = stats.ttest_1samp(OFG_itr, 0)
    OFG_per_ef = OFG_betas[conds[0]] - OFG_betas[conds[1]]
    OFG_per_t, OFG_per_p = stats.ttest_1samp(OFG_per_ef, 0)
    OFG_sem_ef = OFG_betas[conds[2]] - OFG_betas[conds[3]]
    OFG_sem_t, OFG_sem_p = stats.ttest_1samp(OFG_sem_ef, 0)

    print(f'SFG interaction: {t_SFG_itr=:.3f}, {p_SFG_itr=:.4f}')
    print(f'\tSFG perceptual effect: {SFG_per_t=:.3f}, {SFG_per_p=:.4f}')
    print(f'\tSFG semantic effect: {SFG_sem_t=:.3f}, {SFG_sem_p=:.4f}')

    print(f'OFG interaction: {t_OFG_itr=:.3f}, {p_OFG_itr=:.4f}')
    print(f'\tOFG perceptual effect: {OFG_per_t=:.3f}, {OFG_per_p=:.4f}')
    print(f'\tOFG semantic effect: {OFG_sem_t=:.3f}, {OFG_sem_p=:.4f}')

    SFG_x_OFG_itr = SFG_itr - OFG_itr
    t_SFG_x_OFG_itr, p_SFG_x_OFG_itr = stats.ttest_1samp(SFG_x_OFG_itr, 0)
    print(f'Three-way interaction: {t_SFG_x_OFG_itr=:.3f}, {p_SFG_x_OFG_itr=:.4f}')

    sem_itr = SFG_sem_ef - OFG_sem_ef
    t_sem_itr, p_sem_itr = stats.ttest_1samp(sem_itr, 0)
    print(f'Semantic SFG x OFG interaction: {t_sem_itr=:.3f}, {p_sem_itr=:.4f}')

    SFG_betas = plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=big_voxelwise,
                                          get_betas=False, no_lines_stars=True)
    OFG_betas = plot_FigureS1_bars_region('OrG', big_voxelwise=big_voxelwise,
                                          get_betas=False, no_lines_stars=True)
    # print('-*-' * 100)



def plot_SFG_vs_OFG(big_voxelwise=False):
    plot_FigureS1_bars_region('SFG', big_voxelwise=False)
    plot_FigureS1_bars_region('OrG', big_voxelwise=False)

    # plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=True)
    plot_FigureS1_bars_region('SFG', big_voxelwise=True)
    plot_FigureS1_bars_region('OrG', big_voxelwise=True)

if __name__ == '__main__':
    big_voxelwise = True

    # SFG_betas = plot_FigureS1_bars_region('PFC_no_OFC', big_voxelwise=big_voxelwise,
    #                                       get_betas=False, no_lines_stars=True)
    # OFG_betas = plot_FigureS1_bars_region('OrG', big_voxelwise=big_voxelwise,
    #                                       get_betas=False, no_lines_stars=True)
    # quit()

    test_SFG_vs_OFG()
    # plot_SFG_vs_OFG()