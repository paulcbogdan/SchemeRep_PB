import numpy as np
from analyze_rs import sanity_load, get_LSS_SchemeRep
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data0 = sanity_load('102')
    data1 = get_LSS_SchemeRep('102', lsa=True)

    print(data0.shape)
    print(data1.shape)

    rand_voxels = np.random.randint(0, 50, [100, 3])
    rand_voxels += 20

    for (x, y, z) in rand_voxels:
        # print(f'{x=}, {y=}, {z=}')
        l0 = data0[x, y, z, :38]
        if np.sum(np.isnan(l0)):
            continue
        l1 = data1[x, y, z, :38]
        # print(f'{l0=}')
        # print(f'{l1=}')
        r, p = stats.pearsonr(l0, l1)
        # plt.scatter(l0, l1)
        # plt.show()
        # quit()
        print(f'{r=:.3f}')


