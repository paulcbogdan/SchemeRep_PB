import numpy as np

from Utils.atlas_funcs import get_atlas


def matrix_to_csv(mat, fp=r'connRSA_finalizing/DistRep_FC_for_SWD.csv'):
    mat = np.nanmean(mat, axis=0)
    atlas = get_atlas()
    ticks = atlas['ROIs']
    print(mat.shape)
    df = pd.DataFrame(mat, columns=ticks, index=ticks)
    df.to_csv(fp)

