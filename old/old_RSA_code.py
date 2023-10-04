def get_age_str(age):
    return 'healthy' if age == 'healthy' else 'YA' if age == 1 else 'OA'

def get_inc_str(cin):
    return '' if cin is None else \
        '_Con' if cin == 1 else \
        '_Inc' if cin == 2 else \
        '_Neu' if cin == 3 else 'BAD_CIN'


def regress_out_normal_connectivity(mat, age, cin):
    mat = np.array(mat)
    age_str = get_age_str(age)
    inc_str = get_inc_str(cin)
    fp_FC = fr'cache/fCon_{age_str}{inc_str}.pkl'
    FC = pickle_wrap(fp_FC, lambda: get_FC(age, cin),
                     easy_override=False)
    n_ROIs = FC.shape[1]
    for i in range(n_ROIs):
        for j in range(n_ROIs):
            if i == j:
                continue
            print(mat[:, i, j])
            plt.scatter(FC[:, i, j], mat[:, i, j])
            plt.show()
            mat[:, i, j] = regress_out(FC[:, i, j], mat[:, i, j])
            print(mat[:, i, j])
            quit()
    return mat

def plot_test():
    age = 1
    cin = None
    semantic = False
    early_late = True
    fp_out = get_cache_RSA_fp(cin, age, semantic, early_late)
    with open(fp_out, 'rb') as file:
        d = pickle.load(file)

    d_IRAF_conn = d['IRAF_conn']
    mat0 = d_IRAF_conn['obj']
    mat1 = d_IRAF_conn['scn']
    mat2 = d_IRAF_conn['dif']
    mat3 = d_IRAF_conn['dif_']

    fig, axs = plt.subplots(1, 4, figsize=(24, 7))
    # ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs = add_ROI_info()
    plot_connectivity(mat0, ticks, tick_labels, tick_lows, title='obj',
                      ax=axs[0])
    plot_connectivity(mat1, ticks, tick_labels, tick_lows, title='scene',
                      ax=axs[1])
    plot_connectivity(mat2, ticks, tick_labels, tick_lows, title='dif',
                      ax=axs[2])
    plot_connectivity(mat3, ticks, tick_labels, tick_lows, title='dif reg',
                      ax=axs[3])
    plt.tight_layout()
    plt.show()
