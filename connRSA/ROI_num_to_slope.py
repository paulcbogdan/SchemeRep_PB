import utils
from Utils.atlas_funcs import get_BNA_ROIs
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM
from networks.old.networks import prep_networks
from org_sns import get_sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_sn_line(sn, fp, semantic=True, target_ROI='IT'):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    ROIs_match = [ROI for region in regions
                  for ROI in get_BNA_ROIs() if region in ROI]
    ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]
    print(ROI_lvl_control)

    kw = {'semantic': semantic,
          'sn': sn,
          'fp': fp,
          'trial_similarity': 'corr',
          'second_order': 'spear', 'RDM_method': 'within_nan',
          'stdize_by_run': False, 'regress_row': False,
          'return_dif': False
          }

    r_sqs = []
    for i in range(len(ROI_lvl_control)):
        kw['ROI_focus'] = ROI_lvl_control[0]
        kw['ROIs_ctrl'] = ROI_lvl_control[1:i + 1]
        solution, r_sq = do_regr_RSA_sn(**kw)
        r_sqs.append(r_sq)
    return r_sqs
    # print(r_sqs)
    # quit()
    #
    # pass

def test_more_ROIs(region='ITL', semantic=False):
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]

    v_all = []
    for i, sn in tqdm(enumerate(sns)):
        v_sn = []
        for j, fp in enumerate(fps):
            v = utils.pickle_wrap(get_sn_line, kwargs={'sn': sn, 'fp': fp,
                                                       'target_ROI': region,
                                                       'semantic': semantic})
            v_sn.append(v)
        v = np.nanmean(np.array(v_sn), axis=0)
        v_all.append(v)
    v_all = np.array(v_all)
    v_all = np.nanmean(v_all, axis=0)
    plt.title(f'{region} {semantic=}')
    plt.plot(v_all)
    plt.show()



if __name__ == '__main__':
    test_more_ROIs()