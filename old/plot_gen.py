from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import image
from nilearn import plotting

from Utils.atlas_funcs import get_atlas


# def plot_surf(combine_regions=True, bilateral=False):
#     atlas = get_atlas(combine_regions=combine_regions, bilateral=bilateral)

def get_split_cmap(vabs, thresh, cmap, blue_half=False,
                   black_line=0, full_range=False, inferno_half=False):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    if blue_half and black_line > 0:
        n = 4000
    else:
        n = 1000
    prop_gray = thresh / vabs
    if prop_gray > 1:
        prop_gray = 1
        # raise ValueError(f'{prop_black=}')
    n_gray = int(n * prop_gray)
    # vals_black = np.repeat(np.array([0, 0, 0, 1])[:, None], n_black,
    #                      axis=1).T
    vals_gray = np.repeat(np.array([0.5, 0.5, 0.5, 1])[:, None], n_gray,
                          axis=1).T
    prop_colored = 1 - prop_gray

        # vals_black = np.repeat(np.array([0, 0, 0, 1])[:, None], n_black,
        #                        axis=1).T

    n_colored = int(n * prop_colored)
    print(f'{n_colored=}')
    if blue_half:
        vals_colored = cmap(np.linspace(0.05, .9, n_colored))
    else:
        if full_range:
            vals_colored = cmap(np.linspace(0.0, 1., n))
        else:
            vals_colored = cmap(np.linspace(0.05, .9, n))
    #I don't like the purple...
    if inferno_half:
        vals_high = vals_colored
    else:
        vals_low = vals_colored[:int(n_colored/2)]
        vals_high = vals_colored[-int(n_colored/2):]

    if inferno_half:
        cutoff = int(thresh / vabs * len(vals_gray))
        cutoff_flip = len(vals_gray) - cutoff
        vals_gray = vals_gray[:cutoff]
        # print(vals_high.shape)
        # quit()

        x = np.linspace(0, 1, cutoff_flip)
        xp = np.linspace(0, 1, len(vals_high))
        vals_high = [np.interp(x, xp, vals_high[:, i]) for i in range(4)]
        vals_high = np.array(vals_high).T
        # print(vals_high)
        # quit()
        # print(vals_high.shape)
        # quit()
        # vals_high = vals_high[cutoff:]
        # vals_high = vals_high[vals_high.shape[0] // 2:]
        if black_line > 0:
            vals_gray[-1:] = [0, 0, 0, 1]
            vals_high[:1] = [0, 0, 0, 1]
        vals = np.concatenate([vals_gray, vals_high])
    elif blue_half:
        vals_gray = vals_gray[:vals_gray.shape[0] // 4]
        vals_high = vals_high[vals_high.shape[0] // 2:]
        if black_line > 0:
            # n_black = int(n * black_line)
            # print(f'{n_black=}')
            vals_gray[-1:] = [0, 0, 0, 1]
            vals_high[:1] = [0, 0, 0, 1]
        vals = np.concatenate([vals_gray, vals_high])
    else:
        if black_line > 0:
            n_black = int(n * black_line)
            vals_high[:n_black] = [0, 0, 0, 1]
        vals = np.concatenate([vals_low, vals_gray, vals_high])

        # vals = vals[vals.shape[0] // 2:, :]

    # plt.plot(vals[:, 0])
    # plt.show()
    # print(vals[:, 0])
    # quit()

    # print(vals[0:])


    cmap = ListedColormap(vals)

    # print(vals)
    return cmap



def my_plot_surf(Ms, atlas, title, fp_out=None,
                 neg='', pos='', thresh=1.65, vmax=4,
                 cmap='hot_cold', only_positive=False,
                 strict_thresh=True):
    from nilearn import plotting
    atlas['coords'] = np.array(atlas['coords'])
    Ms = np.array(Ms)
    Ms[np.isinf(Ms)] = np.nan
    n_non_nans = np.sum(~np.isnan(Ms))
    img_data = np.zeros(atlas['maps'].shape)
    atlas_data = atlas['maps'].get_fdata()
    # Ms = [np.min([m, 5]) for m in Ms]

    if fp_out is not None:
        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)

    num_above_thresh = np.sum(np.abs(Ms) > thresh)
    for i, val in enumerate(Ms):
        img_data[atlas_data == (i + 1)] = val
        # if val > 1:
        #     print(f'{i} | {val}')
        if i == 102:
            # fix bad part of mesh, FuG appears in Hipp or PhG or w/e
            idxs = np.indices(img_data.shape)
            slicer = idxs[0] > 28
            slicer1 = idxs[0] < 32
            img_data[(atlas_data == (i + 1)) & slicer & slicer1] = 0
        elif i == 103:
            idxs = np.indices(img_data.shape)
            slicer = idxs[0] == 67
            img_data[(atlas_data == (i + 1)) & slicer] = 0

    vabs = np.nanmax(np.abs(Ms))
    # print(f'{vabs=}')
    cmap = get_split_cmap(vabs, thresh, 'rainbow_r',)
    # quit()
    # cmap = 'rainbow_r'
    # if strict_thresh:
    #     img_data[np.abs(img_data) < thresh] = 0
    # else:
    #     img_data[np.abs(img_data) < thresh] = thresh - .1

    img_data[img_data < 0] = 0

    img = image.new_img_like(atlas['maps'], img_data)
    print(f'{thresh=}')
    print(f'{vmax=}')
    fig, axs = plotting.plot_img_on_surf(img, threshold=thresh,
                                         cmap=cmap, title=title,
                                         vmin=0 if only_positive else -vmax,
                                         vmax=vmax,
                                         inflate=False,
                                         surf_mesh='fsaverage5',
                                         avg_method='median')

    # if thresh > 5 or True:
    #     axs[4].set_xticks([-vmax, -thresh, thresh, vmax],
    #                       [-vmax, -thresh, thresh, vmax],
    #                       fontsize=8)
    # elif neg:
    #     axs[4].set_xticks([-vmax, -2, 2, vmax],
    #                       [f'({neg})',
    #                        '-2', '2',
    #                        f'({pos})'], fontsize=8)
    if fp_out is None:
        plotting.show()
    else:
        fig.savefig(fp_out, dpi=300)
        print(f'Saving fig: {fp_out=}')
        plotting.show()

        plt.clf()
    return

    plotting.plot_stat_map(img,
                           vmin=-vmax, vmax=vmax,
                           threshold=thresh, draw_cross=False,
                           display_mode='x',)
                           # cut_coords=[-22, -24, -26, -28])
    plt.show()


if __name__ == '__main__':
    atlas = get_atlas(combine_regions=False)
    L_SFG_coords = []
    L_SFG_idxs = []
    R_IFG_coords = []
    R_IFG_idxs = []
    node_colors = []
    idx = 0
    for i, roi in enumerate(atlas['ROIs']):
        if '_L' in roi and ('SFG' in roi or 'IFG' in roi or 'MFG' in roi or
                            'OrG' in roi or 'ACC' in roi):
            L_SFG_coords.append(atlas['coords'][i])
            node_colors.append('g')
            L_SFG_idxs.append(idx)
            idx += 1
        elif '_R' in roi and ('SFG' in roi or 'IFG' in roi or 'MFG' in roi or
                              'OrG' in roi or 'ACC' in roi):
            R_IFG_coords.append(atlas['coords'][i])
            node_colors.append('g')
            R_IFG_idxs.append(idx)
            idx += 1

    coords = list(L_SFG_coords) + list(R_IFG_coords)
    n_rois = len(coords)
    mat = np.zeros((n_rois, n_rois))
    mesh = np.meshgrid(L_SFG_idxs, R_IFG_idxs)
    mat[mesh] = 1
    mesh = np.meshgrid(R_IFG_idxs, L_SFG_idxs)
    mat[mesh] = 1
    plotting.plot_connectome(mat, coords,
                             edge_threshold=0.1,
                             edge_kwargs={'linewidth': 0.1,
                                          'color': 'k'},
                             node_size=10,
                             node_color=node_colors,
                             # node_kwargs={'c': 'k',
                             #              'size': 10},
                             # title=f'L SFG x R IFG'
                             title = ''
                             )
    plotting.show()