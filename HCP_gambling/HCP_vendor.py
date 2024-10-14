import os
import time
import zlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import image
from scipy import stats as stats
from tqdm import tqdm

from HCP_gambling.preproc_gambling import get_df_events
from atlas_utils import get_atlas
from networks.old.network_funcs import load_FC_for_Lifu
from networks.vendor_partitioning import get_vendor_partitions, do_regression
from old.plot_gen import plot_connectivity
from utils import pickle_wrap


def bar_vendor(conn_highs, conn_lows, combine_regions, bilateral):
    itr, dd, vv, dv_ant, dv_pos, M_overall = get_vd_ef(conn_highs, combine_regions=combine_regions,
                                                       combine_bilateral=bilateral)
    df= pd.DataFrame({'high_PA': dd + vv, 'high_VD': dv_ant + dv_pos,
                       'high_dd': dd, 'high_vv': vv, 'high_dv_ant': dv_ant,
                       'high_dv_pos': dv_pos, })

    itr, dd, vv, dv_ant, dv_pos, M_overall = get_vd_ef(conn_lows, combine_regions=combine_regions,
                                                       combine_bilateral=bilateral)
    # df = df_high.copy()
    df['low_PA'] = dd + vv
    df['low_VD'] = dv_ant + dv_pos
    df['low_dd'] = dd
    df['low_vv'] = vv
    df['low_dv_ant'] = dv_ant
    df['low_dv_pos'] = dv_pos
    # df_low = df[['low_PA', 'low_VD', 'low_dd', 'low_vv', 'low_dv_ant', 'low_dv_pos']]

    for ef in ['PA', 'VD', 'dd', 'vv', 'dv_ant', 'dv_pos']:
        df[f'{ef}_diff'] = df[f'high_{ef}'] - df[f'low_{ef}']
        t, p = stats.ttest_rel(df[f'high_{ef}'], df[f'low_{ef}'])
        N = np.sum(~np.isnan(df[f'high_{ef}']))
        print(f'{ef}: t[{N - 1}] = {t:.2f}, {p=:.4f}')

    # M_PA = (df['high_PA'] + df['low_PA']) / 2
    # df['high_PA'] -= M_PA
    # df['low_PA'] -= M_PA
    # M_VD = (df['high_VD'] + df['low_VD']) / 2
    # df['high_VD'] -= M_VD
    # df['low_VD'] -= M_VD

    # df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['high_VD'].to_list() +
    #                          df['low_PA'].to_list() + df['low_VD'].to_list(),
    #                    'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
    #                    'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2})

    df = pd.DataFrame({'FC': df['high_PA'].to_list() + df['low_PA'].to_list() +
                             df['high_VD'].to_list() + df['low_VD'].to_list(),
                       'PA_VD': ['PA'] * len(df) * 2 + ['VD'] * len(df) * 2,
                       'high_low': (['high'] * len(df) + ['low'] * len(df)) * 2,
                       'sn': list(range(len(df))) * 4})

    plt.rcParams.update({'font.size': 21,
                         'font.sans-serif': 'Arial'})
    # g = sns.catplot(x='high_low', y='FC', hue='PA_VD', data=df,
    #                     kind='bar',
    #                     # errci=68,
    #                     errorbar=('ci', 68),
    #                     # errwidth=1.5,
    #                     edgecolor='k',
    #                     # capsize=0.1, height=4,
    #                     alpha=0.7, linewidth=.7,#.7,
    #                     errwidth=1.2,
    #                     capsize=0.05,
    #                     # palette=sns.color_palette()
    #                     palette=['dodgerblue', 'red'],
    #                     height=5, aspect=0.8
    #                     )
    # plt.ylabel('Mean connectivity')
    # g._legend.remove()
    # g.set_xticklabels(['High PE', 'Low PE'])
    # plt.tight_layout()
    # plt.xlabel('')
    # plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    # plt.xlim(-.5, 1.5)
    # fp = fr'result_pics/other/Study_1B_vendor.png'
    # plt.savefig(fp, dpi=600)
    # plt.show()

    df_PA = df[df['PA_VD'] == 'PA']
    df_PA['FC'] -= df_PA.groupby('sn')['FC'].transform('mean')

    df_VD = df[df['PA_VD'] == 'VD']
    df_VD['FC'] -= df_VD.groupby('sn')['FC'].transform('mean')

    fig, axs = plt.subplots(1, 2, figsize=(6, 5))

    # tips = sns.load_dataset('tips')
    # sns.boxplot(x='day', y='total_bill', data=tips, ax=axs[0])
    # sns.catplot(x='high_low', y='FC', hue='PA_VD', data=df_VD,
    #             ax=axs[0])
    plt.sca(axs[0])
    g = sns.barplot(x='high_low', y='FC', hue='PA_VD', data=df_PA,
                    errorbar=('ci', 68), edgecolor='k',
                    alpha=0.7, linewidth=.7, errwidth=1.2,
                    capsize=0.05, palette=['dodgerblue'],
                    ax=axs[0], legend=False,
                    )
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.xlim(-.5, 1.5)
    plt.ylim(-.023, .023)
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.ylabel('Mean connectivity')
    plt.xlabel('')
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)


    plt.sca(axs[1])
    g = sns.barplot(x='high_low', y='FC', hue='PA_VD', data=df_VD,
                    errorbar=('ci', 68), edgecolor='k',
                    alpha=0.7, linewidth=.7, errwidth=1.2,
                    capsize=0.05, palette=['red'], #height=5,
                    ax=axs[1], legend=False,
                    )
    plt.ylabel('')
    g.set_xticklabels(['High\nPE', 'Low\nPE'])
    plt.xlabel('')
    plt.plot([-.5, 1.5], [0, 0], 'k', linewidth=.5)
    plt.yticks([-.01, 0, .01])
    plt.xlim(-.5, 1.5)
    plt.ylim(-.0115, .0115)
    plt.gca().spines[['bottom', 'top', 'right']].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    fp = fr'result_pics/other/Study_1B_vendor.png'
    # plt.savefig(fp, dpi=600)
    plt.tight_layout()

    plt.show()




def get_sn_roi_ar(sn, lr, combine_regions=False, bilateral=False,
                  reg_global=False, no_compcor=False, rs=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=bilateral,
                      HCP=True)

    glob_str = '_global' if reg_global else ''
    cc_str = '_nocc' if no_compcor else ''
    if rs:
        fp_lsa_lr = fr'E:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
        if not os.path.exists(fp_lsa_lr):
            fp_lsa_lr = fr'C:\HCP_RS_clean\{sn}_REST1_{lr}_clean{glob_str}{cc_str}.nii.gz'
            assert os.path.exists(fp_lsa_lr)
    else:
        fp_lsa_lr = fr'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA\{sn}_{lr}_LSA{glob_str}{cc_str}.nii'
    try:
        img_lsa_lr = image.load_img(fp_lsa_lr)
    except EOFError:
        print('EOFError')
        print(f'{sn=}, {lr=}')
        print(f'{fp_lsa_lr=}')
        raise EOFError
    except zlib.error:
        print('zlib.error')
        print(f'{sn=}, {lr=}')
        print(f'{fp_lsa_lr=}')
        raise zlib.error
    data_lsa_lr = img_lsa_lr.get_fdata()
    # df_events = get_df_events(sn, 'LR')

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    # ROI2vecs = {}
    # region2vecs = defaultdict(list)
    ar = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        roi_data_lsa_lr = data_lsa_lr[atlas_roi]
        vals = roi_data_lsa_lr.mean(axis=0)
        ar.append(vals)
    ar = np.array(ar)
    return ar


def get_conn_sn(sn, combine_regions=False, bilateral=False, drop_neut=False,
                neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
                cont_PE_by_event=False, lr_separate=False,
                reg_global=False, no_compcor=False, median_split=True,
                drop_first=False, both_bhv=False, reset_trial0=False):

    if both_bhv:
        try:
            df_rl, df_lr = get_df_events(sn, 'both', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first, reset_trial0=reset_trial0)
        except Exception as e:
            print(f'ERROR in getting df: {sn}, {e=}')
            # bad_sns.append(sn)
            time.sleep(1)
            return None, sn
    else:
        try:
            df_lr = get_df_events(sn, 'LR', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first,
                                  reset_trial0=reset_trial0)
            df_rl = get_df_events(sn, 'RL', cont_PE=cont_PE,
                                  cont_pe_by_event=cont_PE_by_event,
                                  median_split=median_split,
                                  drop_first=drop_first,
                                  reset_trial0=reset_trial0)
        except Exception as e:
            print(f'ERROR: {sn}, {e=}')
            # bad_sns.append(sn)
            time.sleep(1)
            return None, sn



    try:
        ar = get_sn_roi_ar(sn, 'LR', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=reg_global, no_compcor=no_compcor)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        # bad_sns.append(sn)
        time.sleep(1)
        return None, sn

    if only:
        df_lr.loc[df_lr['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_lr.loc[df_lr['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high = ar[:, df_lr['trial_type'] == 'neut']
    else:
        ar_high = ar[:, df_lr['trial_type'] == 'high_PE']

    ar_low = ar[:, df_lr['trial_type'] == 'low_PE']
    # print(f'{ar_low.shape=}')
    # print(f'{ar_high.shape=}')
    # quit()

    try:
        ar = get_sn_roi_ar(sn, 'RL', combine_regions=combine_regions,
                           bilateral=bilateral, reg_global=reg_global,
                           no_compcor=no_compcor)
    except ValueError:
        print(f'Not analyzed connectivity: {sn}')
        return None, sn
    except Exception as e:
        print(f'ERROR: {sn}, {e=}')
        time.sleep(1)
        # bad_sns.append(sn)
        return None, sn
    if only:
        df_rl.loc[df_rl['event'] != only, 'trial_type'] = 'only'
    if drop_neut or neut_as_PE:
        df_rl.loc[df_rl['event'] == 'neut', 'trial_type'] = 'neut'

    if regr_M:
        ar -= ar.mean(axis=1, keepdims=True)
    if neut_as_PE:
        ar_high2 = ar[:, df_rl['trial_type'] == 'neut']
    else:
        ar_high2 = ar[:, df_rl['trial_type'] == 'high_PE']
    ar_low2 = ar[:, df_rl['trial_type'] == 'low_PE']

    if lr_separate:
        conn_high0 = np.corrcoef(ar_high)
        conn_high0[np.diag_indices_from(conn_high0)] = np.nan
        conn_low0 = np.corrcoef(ar_low)
        conn_low0[np.diag_indices_from(conn_low0)] = np.nan
        conn_high1 = np.corrcoef(ar_high2)
        conn_high1[np.diag_indices_from(conn_high1)] = np.nan
        conn_low1 = np.corrcoef(ar_low2)
        conn_low1[np.diag_indices_from(conn_low1)] = np.nan
        conn_high = (conn_high0 + conn_high1) / 2
        conn_low = (conn_low0 + conn_low1) / 2
    else:
        ar_high = np.concatenate([ar_high, ar_high2], axis=1)
        ar_low = np.concatenate([ar_low, ar_low2], axis=1)
        print(f'{ar_high.shape=} | {ar_low.shape=}')

        conn_high = np.corrcoef(ar_high)
        conn_high[np.diag_indices_from(conn_high)] = np.nan
        conn_low = np.corrcoef(ar_low)
        conn_low[np.diag_indices_from(conn_low)] = np.nan


    # TODO: Lateralized connectivity.
    #  high R-A/high R-P and low L-A/low L-P means A-P connectivity
    return conn_high, conn_low


def make_conn(combine_regions=False, bilateral=False, drop_neut=False,
              neut_as_PE=False, regr_M=True, only=None, cont_PE=None,
              cont_PE_by_event=False, lr_separate=True, num_sns=None,
              n_jobs=1, reg_global=False, no_compcor=False,
              sns_set=None, median_split=True,
              drop_first=False, both_bhv=False, reset_trial0=False,

              sns_final=False):

    if sns_final:
        sns = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101410', '101915', '102008', '102109', '102311', '102513', '102614', '102715', '102816', '103010', '103111', '103212', '103414', '103515', '103818', '104012', '104416', '104820', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '106824', '107018', '107220', '107321', '107422', '107725', '108020', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111211', '111312', '111413', '111514', '111716', '112112', '112314', '112516', '112819', '112920', '113215', '113316', '113417', '113619', '113821', '113922', '114116', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115724', '115825', '116221', '116423', '116524', '116726', '117021', '117122', '117324', '117728', '117930', '118023', '118124', '118225', '118528', '118730', '118831', '118932', '119025', '119126', '119732', '119833', '120010', '120111', '120212', '120414', '120515', '120717', '121315', '121416', '121618', '121719', '121921', '122317', '122418', '122620', '122822', '123117', '123420', '123521', '123723', '123824', '123925', '124220', '124422', '124624', '124826', '125222', '125424', '125525', '126325', '126426', '126628', '127226', '127327', '127630', '127731', '127832', '127933', '128026', '128127', '128329', '128632', '128935', '129028', '129129', '129331', '129634', '129937', '130013', '130114', '130316', '130417', '130518', '130619', '130720', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133625', '133827', '133928', '134021', '134223', '134324', '134425', '134627', '134728', '134829', '135124', '135225', '135528', '135629', '135730', '135932', '136126', '136227', '136631', '136732', '136833', '137027', '137128', '137229', '137431', '137532', '137633', '137936', '138130', '138231', '138332', '138534', '138837', '139233', '139435', '139637', '139839', '140117', '140319', '140824', '140925', '141119', '141422', '141826', '142424', '142828', '143224', '143325', '143426', '143830', '144125', '144226', '144428', '144731', '144832', '144933', '145127', '145531', '145632', '145834', '146129', '146331', '146432', '146533', '146735', '146836', '146937', '147030', '147636', '147737', '148032', '148133', '148335', '148436', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150524', '150625', '150726', '150928', '151021', '151223', '151324', '151425', '151526', '151627', '151728', '151829', '151930', '152225', '152427', '152831', '153025', '153126', '153227', '153429', '153631', '153732', '153833', '153934', '154229', '154330', '154431', '154532', '154734', '154835', '154936', '155231', '155635', '155938', '156031', '156233', '156334', '156435', '156536', '156637', '157336', '157437', '157942', '158035', '158136', '158338', '158540', '158843', '159138', '159239', '159340', '159441', '159744', '159845', '159946', '160123', '160729', '160830', '161327', '161630', '161731', '161832', '162026', '162228', '162329', '162733', '162935', '163129', '163331', '163432', '163836', '164030', '164131', '164636', '164939', '165032', '165234', '165436', '165638', '165840', '165941', '166438', '166640', '167036', '167238', '167440', '167743', '168139', '168240', '168341', '168745', '168947', '169040', '169141', '169343', '169444', '169545', '169747', '169949', '170631', '170934', '171128', '171330', '171431', '171532', '171633', '172029', '172130', '172332', '172433', '172534', '172635', '172938', '173132', '173334', '173435', '173536', '173637', '173738', '173839', '173940', '174437', '174841', '175035', '175136', '175237', '175338', '175439', '175540', '175742', '176037', '176239', '176441', '176542', '176744', '176845', '177140', '177241', '177342', '177645', '177746', '178142', '178243', '178647', '178849', '178950', '179245', '179346', '179952', '180129', '180230', '180432', '180533', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182739', '182840', '183034', '183337', '183741', '185038', '185139', '185341', '185442', '185846', '185947', '186040', '186141', '186444', '186545', '186848', '186949', '187143', '187345', '187547', '187850', '188145', '188347', '188448', '188549', '188751', '189349', '189450', '189652', '190031', '191033', '191235', '191336', '191841', '191942', '192035', '192136', '192237', '192439', '192540', '192641', '192843', '193239', '193441', '193845', '194140', '194443', '194645', '194746', '194847', '195445', '195647', '195849', '195950', '196144', '196346', '196750', '196851', '196952', '197348', '197550', '198047', '198249', '198350', '198653', '198855', '199150', '199251', '199352', '199453', '199655', '199958', '200008', '200109', '200210', '200311', '200513', '200614', '200917', '201111', '201414', '201515', '201717', '201818', '202113', '202719', '202820', '203418', '203923', '204016', '204218', '204319', '204420', '204521', '204622', '205220', '205725', '205826', '206222', '206323', '206525', '206727', '206828', '206929', '207123', '207426', '208024', '208125', '208226', '208327', '209127', '209228', '209329', '209531', '209834', '209935', '210011', '210112', '210415', '210617', '211114', '211215', '211316', '211417', '211619', '211720', '211821', '211922', '212015', '212116', '212217', '212318', '212419', '212823', '213017', '213421', '213522', '214019', '214221', '214524', '214625', '214726', '217126', '217429', '219231', '220721', '221218', '221319', '223929', '224022', '227432', '227533', '228434', '231928', '233326', '236130', '237334', '238033', '239136', '239944', '245333', '246133', '248339', '249947', '250427', '250932', '251833', '255639', '255740', '256540', '257542', '257845', '257946', '268749', '268850', '270332', '274542', '275645', '280739', '280941', '281135', '283543', '284646', '285345', '285446', '286347', '286650', '287248', '289555', '290136', '293748', '295146', '297655', '298051', '298455', '299154', '299760', '300618', '300719', '303119', '303624', '304020', '304727', '305830', '307127', '308129', '308331', '309636', '310621', '311320', '314225', '316633', '316835', '317332', '318637', '320826', '321323', '325129', '329844', '330324', '333330', '334635', '336841', '339847', '341834', '342129', '346137', '346945', '348545', '349244', '350330', '352132', '352738', '353740', '355239', '356948', '358144', '360030', '361234', '361941', '362034', '365343', '366042', '366446', '368551', '368753', '371843', '376247', '377451', '378756', '378857', '379657', '380036', '381038', '381543', '382242', '385046', '385450', '386250', '387959', '389357', '390645', '391748', '392447', '392750', '393247', '393550', '394956', '395251', '395756', '395958', '397154', '397760', '397861', '401422', '406432', '406836', '412528', '413934', '414229', '415837', '419239', '421226', '422632', '424939', '429040', '432332', '433839', '436239', '436845', '441939', '445543', '448347', '449753', '453441', '453542', '454140', '456346', '459453', '461743', '463040', '465852', '467351', '468050', '469961', '473952', '475855', '479762', '480141', '481042', '481951', '485757', '486759', '492754', '495255', '497865', '499566', '500222', '506234', '510225', '510326', '512835', '513130', '513736', '516742', '517239', '518746', '519647', '519950', '520228', '521331', '522434', '523032', '524135', '525541', '529549', '529953', '530635', '531536', '536647', '540436', '541640', '541943', '545345', '548250', '549757', '550439', '552241', '552544', '553344', '555348', '555651', '555954', '557857', '558657', '558960', '559053', '559457', '561242', '561444', '561949', '562345', '562446', '565452', '566454', '567052', '567759', '567961', '568963', '570243', '571144', '572045', '573249', '573451', '576255', '578057', '578158', '579665', '579867', '580044', '580347', '580650', '580751', '581349', '581450', '583858', '585256', '585862', '586460', '587664', '588565', '589567', '590047', '592455', '594156', '597869', '598568', '599065', '599469', '599671', '601127', '604537', '609143', '611938', '613235', '613538', '614439', '615441', '615744', '616645', '617748', '618952', '620434', '622236', '623137', '623844', '626648', '627549', '627852', '628248', '633847', '634748', '635245', '638049', '644044', '644246', '645450', '645551', '647858', '654350', '654552', '654754', '656253', '656657', '657659', '660951', '662551', '663755', '664757', '665254', '667056', '668361', '671855', '672756', '673455', '675661', '677766', '677968', '679568', '679770', '680250', '680452', '680957', '683256', '685058', '686969', '687163', '688569', '689470', '690152', '692964', '693764', '694362', '695768', '698168', '700634', '701535', '702133', '704238', '705341', '706040', '707749', '709551', '715041', '715647', '715950', '720337', '723141', '724446', '725751', '727553', '727654', '728454', '729254', '729557', '731140', '732243', '734045', '734247', '735148', '737960', '742549', '744553', '748258', '748662', '749058', '749361', '751348', '751550', '753150', '753251', '756055', '757764', '759869', '760551', '761957', '763557', '765056', '765864', '766563', '767464', '769064', '770352', '771354', '773257', '774663', '779370', '782561', '783462', '784565', '786569', '788674', '788876', '789373', '792564', '792867', '793465', '800941', '802844', '803240', '804646', '809252', '810439', '810843', '814548', '814649', '815247', '816653', '818455', '818859', '820745', '822244', '825048', '825553', '825654', '826353', '826454', '827052', '828862', '832651', '833148', '833249', '835657', '837560', '837964', '841349', '843151', '844961', '845458', '849264', '849971', '852455', '856463', '856766', '856968', '857263', '861456', '865363', '867468', '869472', '870861', '871762', '871964', '872562', '872764', '873968', '877168', '877269', '878776', '878877', '880157', '882161', '884064', '885975', '886674', '887373', '888678', '889579', '891667', '894067', '894673', '894774', '896778', '896879', '898176', '899885', '901038', '901139', '901442']
    else:
        fns = os.listdir(r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSA')
        if reg_global:
            fns = [fn for fn in fns if 'global' in fn]
        else:
            fns = [fn for fn in fns if 'global' not in fn]
        if no_compcor:
            fns = [fn for fn in fns if 'nocc' in fn]
        else:
            fns = [fn for fn in fns if 'nocc' not in fn]
        sns = {fn.split('_')[0] for fn in fns}
        sns = sorted(list(sns))
        print(f'{len(sns)=}')
        if sns_set is not None:
            sns = [sn for sn in sns if sn in sns_set]
        if num_sns is None:
            num_sns = 10_000
    # if sns is not None:
    #     sns = sns[:num_sns]
    # sns = sns[:-1]
    # sns = sns[::-1]

    conn_highs = []
    conn_lows = []
    bad_sns = []
    kw = {'combine_regions': combine_regions,  'bilateral': bilateral,
          'neut_as_PE': neut_as_PE, 'drop_neut': drop_neut, 'regr_M': regr_M,
          'only': only, 'cont_PE': cont_PE, 'cont_PE_by_event': cont_PE_by_event,
          'lr_separate': lr_separate, 'reg_global': reg_global,
          'no_compcor': no_compcor, 'median_split': median_split,
          'drop_first': drop_first, 'both_bhv': both_bhv,
          'reset_trial0': reset_trial0}

    if not no_compcor:
        del kw['no_compcor']

    # if no_compcor:
    #     from datetime import datetime
    #     dt_max = datetime(2024, 9, 15, 19, 50, 0)
    # elif cont_PE_by_event:
    #     from datetime import datetime
    #     dt_max = datetime(2024, 9, 15, 11, 0, 0)
    # else:
    #     dt_max = None

    from datetime import datetime
    dt_max = datetime(2024, 9, 21, 18, 45, 0)

    good_sns = []
    while len(good_sns) < num_sns and (len(sns) > 0):#, total=num_sns):
        sn = sns.pop()
        kw['sn'] = sn
        conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                             easy_override=False,
                                             dt_max=dt_max)
        if conn_high is None:
            print(f'Bad conn: {sn}, attempting to redo')
            conn_high, conn_low_sn = pickle_wrap(get_conn_sn, kwargs=kw,
                                                 easy_override=True,
                                                 dt_max=dt_max)
        if conn_high is None:
            print('BAD CONN??')
            bad_sns.append(sn)
            continue

        conn_highs.append(conn_high)
        conn_lows.append(conn_low_sn)
        good_sns.append(sn)

    print(f'{bad_sns=}')
    conn_highs = np.array(conn_highs)
    conn_lows = np.array(conn_lows)

    print(f'Final sns: {len(good_sns)=}')
    return conn_highs, conn_lows, good_sns


def get_vd_ef(conn, combine_regions=False, combine_bilateral=False,
              anat_ver=3):
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', anat=True, weighted=False,
                              flip=True, thr=.9, scrub=False, anat_ver=anat_ver,
                              combine_regions=combine_regions)
    # print(f'{p_d_ant=},\n{p_d_pos=},\n{p_v_ant=},\n{p_v_pos=}')

    if combine_bilateral:
        p_d_ant = np.array(p_d_ant[::2]) // 2
        p_d_pos = np.array(p_d_pos[::2]) // 2
        p_v_ant = np.array(p_v_ant[::2]) // 2
        p_v_pos = np.array(p_v_pos[::2]) // 2

    dd = conn[:, *np.ix_(p_d_pos, p_d_ant)]
    dd = np.nanmean(dd, axis=(1, 2))
    vv = conn[:, *np.ix_(p_v_pos, p_v_ant)]
    vv = np.nanmean(vv, axis=(1, 2))
    dv_ant = conn[:, *np.ix_(p_d_ant, p_v_ant)]
    dv_ant = np.nanmean(dv_ant, axis=(1, 2))
    dv_pos = conn[:, *np.ix_(p_d_pos, p_v_pos)]
    dv_pos = np.nanmean(dv_pos, axis=(1, 2))
    M_overall = np.nanmean(conn, axis=(1, 2))


    return dd + vv - dv_ant - dv_pos, dd, vv, dv_ant, dv_pos, M_overall


def get_combo(kw):
    kw['only'] = 'loss'
    conn_highs, conn_lows, sns = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    kw['only'] = 'win'
    conn_highs2, conn_lows2, sns2 = (
        pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    assert sns == sns2
    conn_highs = np.mean([conn_highs, conn_highs2], axis=0)
    conn_lows = np.mean([conn_lows, conn_lows2], axis=0)
    return conn_highs, conn_lows, sns


def test_vendor(combine_regions=False, bilateral=False, corr_z=True,
                sub_ROI_expected=False):

    kw = {'combine_regions': combine_regions, 'bilateral': False, 'neut_as_PE': None,
          'drop_neut': True, 'only': 'combo', 'num_sns': 1000, 'cont_PE': 0.5,
          'cont_PE_by_event': True,
          'regr_M': True, 'lr_separate': True,
          'reg_global': True, 'no_compcor': True, 'median_split': True,
          'drop_first': False, 'both_bhv': True, 'reset_trial0': True,

          'sns_final': True}

    # kw = {'combine_regions': combine_regions, 'bilateral': False, 'neut_as_PE': None,
    #       'drop_neut': True, 'only': 'combo', 'num_sns': 1000, 'cont_PE': 0.3,
    #       'cont_PE_by_event': True,
    #       'regr_M': True, 'lr_separate': True,
    #       'reg_global': True, 'no_compcor': True, 'median_split': True,
    #       'drop_first': False, 'both_bhv': True, 'reset_trial0': True,
    #
    #       'sns_final': True}

    # TODO: align 1000 subjects to be same across rs-fMRI and task-fMRI

    print(f'{kw=}')

    if kw['neut_as_PE']:
        kw['drop_neut'] = False
        kw['only'] = None
        kw['cont_PE'] = None
        kw['cont_PE_by_event'] = False

    if kw['only'] == 'combo':
        conn_highs, conn_lows, sns = get_combo(kw)
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))


    if combine_regions:
        conn_highs[:, :, 46:] = np.nan
        conn_highs[:, 46:, :] = np.nan
        conn_lows[:, :, 46:] = np.nan
        conn_lows[:, 46:, :] = np.nan
    else:
        conn_highs[:, :, 210:] = np.nan
        conn_highs[:, 210:, :] = np.nan
        conn_lows[:, :, 210:] = np.nan
        conn_lows[:, 210:, :] = np.nan

    if sub_ROI_expected:
        ROI_expected = np.nanmean(conn_highs, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_highs -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_highs /= ROI_expected[None]
        ROI_expected = np.nanmean(conn_lows, axis=(0, 2))
        ROI_expected = (ROI_expected[:, None] + ROI_expected[None, :]) / 2
        conn_lows -= ROI_expected[None]
        # ROI_expected = np.sqrt(ROI_expected[:, None] * ROI_expected[None, :])
        # conn_lows /= ROI_expected[None]

    bar_vendor(conn_highs, conn_lows, combine_regions, bilateral)

    overall = (conn_highs + conn_lows) / 2
    print(f'{overall.shape=}')

    plt.rcParams.update({'font.size': 16,
                         'font.sans-serif': 'Arial'})
    # M = np.nanmean(overall, axis=0)
    # M_flat = M.flatten()
    # plt.hist(M_flat, bins=100)
    # plt.title('HCP FC histogram')
    # plt.show()
    # quit()

    dif = conn_highs - conn_lows
    M = np.nanmean(dif, axis=0)
    SE = stats.sem(dif, axis=0, nan_policy='omit')
    t = M / SE

    print(t.shape)


    if corr_z:
        t_flat = t[np.tril_indices_from(t, k=-1)]
        z_both = get_SchemeRep_regr(combine_regions=combine_regions, plot=False)
        # print(z_both.shape)
        # quit()
        z_flat = z_both[np.tril_indices_from(z_both, k=-1)]
        r, p = stats.spearmanr(t_flat, z_flat, nan_policy='omit')
        print(f'Gambling x SchemeRep: {r=:.2f}, {p=:.3f}')
    else:
        r = None


        # quit()

    ef_high = get_vd_ef(conn_highs, combine_regions=combine_regions,
                        combine_bilateral=bilateral)[0]
    ef_low = get_vd_ef(conn_lows, combine_regions=combine_regions,
                       combine_bilateral=bilateral)[0]
    itr = ef_low - ef_high
    t_final, p = stats.ttest_1samp(itr, 0)
    # plt.hist(itr)
    # plt.show()
    # quit()
    N = itr.shape[0]
    nans = np.sum(np.isnan(itr))
    F = t_final ** 2
    print(f't[{N - nans - 1}/{N - 1}] = {t_final:.2f}, {p=:.4f}, F = {F:.2f}')
    print(kw)



    if not bilateral:
        atlas = get_atlas(combine_regions=combine_regions,
                          combine_bilateral=bilateral, HCP=True,
                          lifu_labels=combine_regions)


        title = str(kw)
        title_ = ''
        for i in range(len(title) // 50):
            title_ += title[i * 50:(i + 1) * 50] + '\n'
        title_ += title[(i + 1) * 50:]
        title = title_

        title += f'\nt[{N - nans - 1}/{N - 1}] = {t_final:.2f}'
        if r is not None:
            title += f', {r=:.2f}'
        # quit()

        # p_v_pos = [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
        # p_d_pos = [134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145]
        # plt.imshow(t[np.ix_(p_v_pos, p_d_pos)])
        # plt.colorbar()
        # plt.show()
        # t[np.abs(t) < 3] = np.nan
        M_high = np.nanmean(conn_highs, axis=0)
        M_low = np.nanmean(conn_lows, axis=0)
        # print(len(atlas['tick_labels']))
        # print(len(atlas['ticks']))
        # quit()
        if combine_regions:
            atlas['ticks'] = atlas['ticks'][:23]
            atlas['tick_labels'] = atlas['tick_labels'][:23]
            atlas['tick_lows'] = atlas['tick_lows'][:23]
            t = t[:46, :46]
            fp_out = r'C:\PycharmProjects\SchemeRep\result_pics\other\Study_1B_PE_matrix.png'
        else:
            fp_out = None

        plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title=title, tile=.01,
                          no_avg=True, cbar_label='t-value',
                          vmin=-6, vmax=6, fp=fp_out)

        # plot_connectivity(M_high, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # plot_connectivity(M_low, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # plot_connectivity(M_high - M_low, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   )
        # quit()

        # z_threshed = z_both
        # z_threshed[np.abs(z_threshed) < 2] = np.nan
        #
        # z_threshed[np.abs(t) < 3] = np.nan
        # plot_connectivity(z_threshed, atlas['ticks'], atlas['tick_labels'],
        #                   atlas['tick_lows'], title=title, tile=.01,
        #                   no_avg=True, cbar_label='Correlation (r)',
        #                   vmin=-4, vmax=4)
        # conjunct = np.logical_and(np.abs(t) > 3, np.abs(z_both) > 2)

    quit()
    n, bins, patches = plt.hist(itr, range=(-0.4, 0.4), bins=40)
    plt.plot([0, 0], [0, np.max(n)], 'r--')
    plt.show()

    plt.hist(itr, range=(-0.4, 0.4), bins=40,
             cumulative=True, density=True)
    plt.plot([0, 0], [0, 1], 'r--')
    plt.plot([-.4, .4], [0.5, 0.5], 'r--')
    plt.xlim(-0.4, 0.4)
    plt.ylim(0, 1)
    plt.show()
    quit()


def get_SchemeRep_regr(regress=False, combine_regions=False, plot=False):
    kwargs = {'fp': 'obj7_fMRI',
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')

    z_both = do_regression(sn_inc_conn, flip=False) # False = (Incongruent > Congruent)

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False, HCP=True,
                      lifu_labels=True)
    if combine_regions:
        z_both = z_both[:54, :54]
    if plot:
        plot_connectivity(z_both, atlas['ticks'], atlas['tick_labels'],
                          atlas['tick_lows'], title='SchemeRep matrix', tile=.01,
                          no_avg=True, cbar_label='Correlation (r)')

    return z_both

if __name__ == '__main__':
    # get_SchemeRep_regr(combine_regions=False, plot=True)
    # test_corr()
    # fp = r'C:\PycharmProjects\SchemeRep\HCP_gambling\LSS\100206_LR_LSS.nii'
    # img = image.load_img(fp)
    # print(img.shape)
    # LSS_gambling()
    test_vendor()