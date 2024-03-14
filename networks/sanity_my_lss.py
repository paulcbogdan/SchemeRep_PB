import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')
import numpy as np
from load_more import get_LSS_SchemeRep, sanity_load
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# What is the onset time rleative to?

if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)

    data0 = sanity_load('103')
    data1 = get_LSS_SchemeRep('103', lsa=False)
    # data0 = get_LSS_SchemeRep('102', lsa=False)

    print(data0.shape)
    print(data1.shape)

    rand_voxels = np.random.randint(0, 50, [100, 3])
    rand_voxels += 20

    M_rs = []
    for (x, y, z) in rand_voxels:
        # print(f'{x=}, {y=}, {z=}')
        l0 = data0[x, y, z, :38]
        if np.sum(np.isnan(l0)):
            continue
        l1 = data1[x, y, z, :38]
        # print(f'{l0=}')
        # print(f'{l1=}')
        r, p = stats.spearmanr(l0, l1)
        # plt.scatter(l0, l1)
        # plt.show()
        # quit()
        print(f'{r=:.3f}')
        M_rs.append(r)
    M_rs = np.array(M_rs)
    print(f'{np.nanmean(M_rs)=:.3f}')


