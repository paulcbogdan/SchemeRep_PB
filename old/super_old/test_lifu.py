import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as io

from stim import get_img_fns


def get_stim_RDM_lifu(df_sn):
    print('Loading existing...')
    fp_in = r'C:\PycharmProjects_C\SchemeRep\RSAmodels\example_deepNeuralNetworkScripts_from_Lifu\RSAmodel\modelRDMs' \
            r'\RSM_VGG16_PCA.mat'
    mat = io.loadmat(fp_in)
    RDM_stim = mat['R']
    RDM_new = np.zeros((len(df_sn), len(df_sn)))

    tblStim = pd.read_csv(r"SchemRep_tasks\PTBtasks\fullStimList.csv")
    tblStim.head()
    filelist = tblStim['ObjectFile'].to_list()
    name2fps, _ = get_img_fns(get_dict=True)

    for obj0 in tqdm(df_sn['obj'], desc='prepping Lifu RDM'):
        obj0 = name2fps[obj0].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
        for obj1 in df_sn['obj']:
            obj1 = name2fps[obj1].replace(r'SchemRep_tasks\PTBtasks\updatedObjectsResampled', '')[1:]
            idx0 = filelist.index(obj0)
            idx1 = filelist.index(obj1)
            RDM_new[idx0, idx1] = RDM_stim[idx0, idx1]
            RDM_new[idx1, idx0] = RDM_stim[idx1, idx0]
    # pd.DataFrame(RDM_new).to_csv('RDM_stim_lifu.csv')

    return RDM_new
