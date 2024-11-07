import os
import pickle
from pathlib import Path

import numpy as np
from nichord.combine import plot_and_combine

from Utils.pickle_wrap_funcs import pickle_wrap


def get_binary_matrix(matrix, threshold=.9, rowwise=True, intersect=False):
    if rowwise:
        matrix_mask = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            row = matrix[i]
            median = np.nanquantile(row, threshold)
            matrix_mask[i, row > median - .0001] += 0.85 if intersect else 1
            matrix_mask[row > median - .0001, i] += 0.85 if intersect else 1
        matrix_mask[matrix_mask > 1.5] = 1
        matrix_binary = matrix.copy()
        matrix_binary[matrix_mask < 1] = 0
    else:
        if threshold < 1:
            threshold = np.nanquantile(matrix, threshold)
        print(f'Binary threshold: {threshold=:.3f}')
        matrix_binary = matrix.copy()
        matrix_mask = matrix > threshold
        matrix_binary[matrix_binary < threshold] = 0
        matrix_binary[matrix_binary >= threshold] = 1
        matrix[matrix < threshold] = 0

    return matrix_binary, matrix_mask


def get_modules(matrix_thresh):
    import leidenalg
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(matrix_thresh)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=0)
    return part


def get_partition_matrix(mat, idx, w_zeros=False):
    if w_zeros:
        mat_new = np.zeros(mat.shape)
        mat_new[np.ix_(idx, idx)] = mat[np.ix_(idx, idx)]
        return mat_new
    else:
        meshy = np.ix_(idx, idx)
        slicer = tuple([slice(None)] * (mat.ndim - 2) + [meshy[0], meshy[1]])
        return mat[slicer]


def plot_nichord(coords, fn, title, dir_out='nichord_plots',
                 corr=None, edges=None, edge_weights=None):
    from nichord.convert import convert_matrix
    from nichord.coord_labeler import get_idx_to_label
    if edges is None and edge_weights is None:
        assert corr is not None, 'Either corr or edges and edge_weights must ' \
                                 'be provided'
        edges, edge_weights = convert_matrix(corr)

    search_closest = False
    fp_idx_to_label = fr'cache/idx_to_label_{len(coords)}_{search_closest}.pkl'

    idx_to_label = pickle_wrap(lambda: get_idx_to_label(
        coords, atlas='yeo', search_closest=search_closest),
                               fp_idx_to_label, easy_override=True, )

    if len(edge_weights) == 0:
        print('Zero edge weights!')
        return

    chord_kw = {'alphas': .5}
    glass_kw = {}

    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    if 'ttest_modules' in dir_out or 'Fig2' in dir_out:
        network_colors = {'Uncertain': 'k', 'Visual': 'k',
                          'SM': 'k', 'DAN': 'k', 'VAN': 'k',
                          'Limbic': 'k', 'FPCN': 'k', 'DMN': 'k'}
        chord_kw['vmin'] = -3.5
        chord_kw['vmax'] = 3.5
        glass_kw['vmin'] = -3.5
        glass_kw['vmax'] = 3.5
        glass_kw['linewidths'] = 3.
        if 'PA_' in dir_out:
            chord_kw['vmax'] = 2
            glass_kw['vmax'] = 2
            edge_weights = -edge_weights
        else:
            chord_kw['vmin'] = -2
            glass_kw['vmin'] = -2
        # glass_kw['node_size']

    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    print('Plotting and combining NiChord')
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=coords,
                     network_order=network_order, network_colors=network_colors,
                     title=title, chord_kwargs=chord_kw,
                     glass_kwargs=glass_kw,
                     only1glass=False)


def get_BNA_coords(code='BNA'):
    if code == 'BNA':
        coords = [[-5, 15, 54], [7, 16, 54], [-18, 24, 53], [22, 26, 51], [-11, 49, 40], [13, 48, 40], [-18, -1, 65],
                  [20, 4, 64], [-6, -5, 58], [7, -4, 60], [-5, 36, 38], [6, 38, 35], [-8, 56, 15], [8, 58, 13],
                  [-27, 43, 31], [30, 37, 36], [-42, 13, 36], [42, 11, 39], [-28, 56, 12], [28, 55, 17], [-41, 41, 16],
                  [42, 44, 14], [-33, 23, 45], [42, 27, 39], [-32, 4, 55], [34, 8, 54], [-26, 60, -6], [25, 61, -4],
                  [-46, 13, 24], [45, 16, 25], [-47, 32, 14], [48, 35, 13], [-53, 23, 11], [54, 24, 12], [-49, 36, -3],
                  [51, 36, -1], [-39, 23, 4], [42, 22, 3], [-52, 13, 6], [54, 14, 11], [-7, 54, -7], [6, 47, -7],
                  [-36, 33, -16], [40, 39, -14], [-23, 38, -18], [23, 36, -18], [-6, 52, -19], [6, 57, -16],
                  [-10, 18, -19], [9, 20, -19], [-41, 32, -9], [42, 31, -9], [-49, -8, 39], [55, -2, 33], [-32, -9, 58],
                  [33, -7, 57], [-26, -25, 63], [34, -19, 59], [-13, -20, 73], [15, -22, 71], [-52, 0, 8], [54, 4, 9],
                  [-49, 5, 30], [51, 7, 30], [-8, -38, 58], [10, -34, 54], [-4, -23, 61], [5, -21, 61], [-32, 14, -34],
                  [31, 15, -34], [-54, -32, 12], [54, -24, 11], [-50, -11, 1], [51, -4, -1], [-62, -33, 7],
                  [66, -20, 6], [-45, 11, -20], [47, 12, -20], [-55, -3, -10], [56, -12, -5], [-65, -30, -12],
                  [65, -29, -13], [-53, 2, -30], [51, 6, -32], [-59, -58, 4], [60, -53, 3], [-58, -20, -9],
                  [58, -16, -10], [-45, -26, -27], [46, -14, -33], [-51, -57, -15], [53, -52, -18], [-43, -2, -41],
                  [40, 0, -43], [-56, -16, -28], [55, -11, -32], [-55, -60, -6], [54, -57, -8], [-59, -42, -16],
                  [61, -40, -17], [-55, -31, -27], [54, -31, -26], [-33, -16, -32], [33, -15, -34], [-31, -64, -14],
                  [31, -62, -14], [-42, -51, -17], [43, -49, -19], [-27, -7, -34], [28, -8, -33], [-25, -25, -26],
                  [26, -23, -27], [-28, -32, -18], [30, -30, -18], [-19, -12, -30], [19, -10, -30], [-23, 2, -32],
                  [22, 1, -36], [-17, -39, -10], [19, -36, -11], [-54, -40, 4], [53, -37, 3], [-52, -50, 11],
                  [57, -40, 12], [-16, -60, 63], [19, -57, 65], [-15, -71, 52], [19, -69, 54], [-33, -47, 50],
                  [35, -42, 54], [-22, -47, 65], [23, -43, 67], [-27, -59, 54], [31, -54, 53], [-34, -80, 29],
                  [45, -71, 20], [-38, -61, 46], [39, -65, 44], [-51, -33, 42], [47, -35, 45], [-56, -49, 38],
                  [57, -44, 38], [-47, -65, 26], [53, -54, 25], [-53, -31, 23], [55, -26, 26], [-5, -63, 51],
                  [6, -65, 51], [-8, -47, 57], [7, -47, 58], [-12, -67, 25], [16, -64, 25], [-6, -55, 34], [6, -54, 35],
                  [-50, -16, 43], [50, -14, 44], [-56, -14, 16], [56, -10, 15], [-46, -30, 50], [48, -24, 48],
                  [-21, -35, 68], [20, -33, 69], [-36, -20, 10], [37, -18, 8], [-32, 14, -13], [33, 14, -13],
                  [-34, 18, 1], [36, 18, 1], [-38, -4, -9], [39, -2, -9], [-38, -8, 8], [39, -7, 8], [-38, 5, 5],
                  [38, 5, 5], [-4, -39, 31], [4, -37, 32], [-3, 8, 25], [5, 22, 12], [-6, 34, 21], [5, 28, 27],
                  [-8, -47, 10], [9, -44, 11], [-5, 7, 37], [4, 6, 38], [-7, -23, 41], [6, -20, 40], [-4, 39, -2],
                  [5, 41, 6], [-11, -82, -11], [10, -85, -9], [-5, -81, 10], [7, -76, 11], [-6, -94, 1], [8, -90, 12],
                  [-17, -60, -6], [18, -60, -7], [-13, -68, 12], [15, -63, 12], [-31, -89, 11], [34, -86, 11],
                  [-46, -74, 3], [48, -70, -1], [-18, -99, 2], [22, -97, 4], [-30, -88, -12], [32, -85, -12],
                  [-11, -88, 31], [16, -85, 34], [-22, -77, 36], [29, -75, 36], [-19, -2, -20], [19, -2, -19],
                  [-27, -4, -20], [28, -3, -20], [-22, -14, -19], [22, -12, -20], [-28, -30, -10], [29, -27, -10],
                  [-12, 14, 0], [15, 14, -2], [-22, -2, 4], [22, -2, 3], [-17, 3, -9], [15, 8, -9], [-23, 7, -4],
                  [22, 8, -1], [-14, 2, 16], [14, 5, 14], [-28, -5, 2], [29, -3, 1], [-7, -12, 5], [7, -11, 6],
                  [-18, -13, 3], [12, -14, 1], [-18, -23, 4], [18, -22, 3], [-7, -14, 7], [3, -13, 5], [-16, -24, 6],
                  [15, -25, 6], [-15, -28, 4], [13, -27, 8], [-12, -22, 13], [10, -14, 14], [-11, -14, 2], [13, -16, 7]]
    elif code == 'schaefer':
        with open(f'cache/schaefer_coords.pkl', 'rb') as f:
            coords = pickle.load(f)
    else:
        raise NotImplementedError(f'Unknown code: {code}')
    return coords


def get_main_partitions(sn_inc_conn, coords=None, plot=False,
                        threshold=.95, dir_out=None, dir_out_full=None, ):
    M_conn = np.nanmean(sn_inc_conn,
                        axis=tuple(range(len(sn_inc_conn.shape[:-2]))))

    matrix_binary, matrix_mask = get_binary_matrix(M_conn,
                                                   threshold=threshold)

    matrix_binary[np.isnan(matrix_binary)] = 0
    M_conn_masked = M_conn * matrix_mask

    partitions = get_modules(matrix_binary)

    if plot:
        plot_partitions(partitions, M_conn_masked, coords=coords,
                        dir_out=dir_out, dir_out_full=dir_out_full, )

    partitions = [p for p in partitions]
    return partitions, matrix_mask


def plot_partitions(partitions, M_conn_masked, coords=None,
                    dir_out=None, dir_out_full=None, title_extra=''):
    if coords is None:
        coords = get_BNA_coords()
    for i, p in enumerate(partitions):
        if len(p) < 5:
            continue
        M_corr_part = get_partition_matrix(M_conn_masked, p, w_zeros=True)
        assert M_corr_part.shape in [(246, 246), (54, 54)]
        fn = f'module_{i}.png'
        title = f'Partition {i + 1}{title_extra}'
        cur_dir = os.getcwd()
        if dir_out_full:
            dir_out_ = dir_out_full
        else:
            dir_out_ = f'{cur_dir}/result_pics/nichord/{dir_out}'

        print(f'Plotting partition: {i} | {p=}, {fn=}')
        print(f'\t{dir_out_=}')
        plot_nichord(coords, fn, title, corr=M_corr_part,
                     dir_out=dir_out_, )


def get_FC_between_ROIs(conn_trials, p_mod0, p_mod1, trialwise=True,
                        transpose=True):
    if transpose:
        conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))

    meshy = np.ix_(p_mod0, p_mod1)
    slicer = tuple([slice(None)] * (conn_trials.ndim - 2) + [meshy[0], meshy[1]])
    conn_trials_cross = conn_trials[slicer]

    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    if trialwise:
        agg_zs = np.nanmean(flat_cross, axis=-1)
        agg_zs = np.nanmean(agg_zs, axis=1)  # omit inc axis
        return agg_zs
    else:
        rs = np.nanmean(flat_cross, axis=2)
        agg_rs = np.nanmean(rs, axis=-1)
        return agg_rs
