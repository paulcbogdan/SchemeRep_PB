from time import time

t_st = time()

from Utils.atlas_funcs import get_atlas
import numpy as np
import matplotlib.pyplot as plt

from nilearn import plotting, image


def plot_t(t, title, vabs=None, fn='', only_positive=True, flip_color=False,
           dic='searchlight', thresh=2.40):
    # t[59] = 2.65, p = .0052
    # t[59] = 2.40, p = .0098

    t_M = np.nanmean(t)


    print(f'{t_M=:.3f}')
    t_min = np.nanmin(t)
    print(f'\t{t_min=:.2f}')
    t_max = np.nanmax(t)
    print(f'\t{t_max=:.2f}')

    atlas = get_atlas()

    x_pre_pad = atlas['maps'].shape[0] - t.shape[0]
    y_pre_pad = atlas['maps'].shape[1] - t.shape[1]
    y_post_pad = y_pre_pad // 2
    y_pre_pad -= y_post_pad
    z_pre_pad = atlas['maps'].shape[2] - t.shape[2]
    t = np.pad(t, ((x_pre_pad, 0), (y_pre_pad, y_post_pad), (z_pre_pad, 0)))
    # t = mask_img(t, None, blocks=True)

    if only_positive:
        t = np.maximum(t, 0)
    t_img = image.new_img_like(atlas['maps'], t)

    t_img = image.threshold_img(t_img, thresh)

    ROIs = atlas['ROIs']
    ROI_nums = atlas['ROI_nums']
    ROI_regions = atlas['ROI_regions']
    l_Ms = []
    for j, (ROI, ROI_num, region) in enumerate(zip(ROIs, ROI_nums, ROI_regions)):
        atlas_roi = atlas['maps'].get_fdata() == ROI_num
        t_roi = t[atlas_roi]
        M_r_M = np.nanmean(t_roi)
        l_Ms.append(M_r_M)


    # t_img = image.threshold_img(t_img, threshold=3,
    #                             cluster_threshold=40)
    if vabs is None:
        vabs = np.nanquantile(np.abs(t), 0.995)
    print(f'{vabs=}')
    # quit()

    cmap = 'inferno' if only_positive else 'turbo'
    if flip_color:
        title = title.replace('Blue', 'XXX')
        title = title.replace('Red', 'Blue')
        title = title.replace('XXX', 'Red')
        cmap += '_r'


    d_clean = 'cleanlight'
    fp_clean = fr'C:\PycharmProjects\SchemeRep\result_pics\{d_clean}\{fn}.png'
    fig = plt.figure(figsize=(8, 4.5))
    full_sns = '60,' in fn

    plotting.plot_glass_brain(t_img, vmin=thresh if only_positive else -vabs,
                              display_mode='xz', vmax=vabs, plot_abs=False,
                              threshold=.001, cmap=cmap, annotate=False,
                              resampling_interpolation='nearest',
                              title = None if full_sns else title,
                              figure=fig)
    plt.savefig(fp_clean, dpi=300)
    plt.show()
