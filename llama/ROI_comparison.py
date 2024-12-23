import numpy as np
from sympy.physics.vector import cross

from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from llama.model_settings import get_base_kw, get_explore_llama, get_explore_BERT
from llama.run_many_layers import run_layer
from old.plot_gen import my_plot_surf
from scipy import stats

def process_allow_misses(kw, ROI):
    if ROI in ['69 ATL_L_6_1', '70 ATL_R_6_1', '89 ITG_L_7_1',
               '96 ITG_R_7_4', '101 ITG_L_7_7', '102 ITG_R_7_7',
               ]:
        kw['allow_misses'] = ('131',)
    elif ROI in ['109 PhG_L_6_1', '110 PhG_R_6_1', '111 PhG_L_6_2',
                 '112 PhG_R_6_2', '116 PhG_R_6_4', '117 PhG_L_6_5',
                 '118 PhG_R_6_5']:
        kw['allow_misses'] = ('131', '135', '235')
    elif ROI in ['93 ATL_L_7_3', '94 ATL_R_7_3']:
        kw['allow_misses'] = 'all'
        # kw['allow_misses'] = ('111', '131', '135', '218', '235')
    elif ROI in ['233 Tha_L_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3']:
        kw['allow_misses'] = 'all'
    else:
        kw['allow_misses'] = None


def item_vs_attn_ROIs(combine_regions=False, st=8, end=None,
                      activation_model='meta-llama/Llama-3.2-3b'):
    llama31_3b_attn = get_explore_llama(activation_model,
                                        attn=True, st=st, end=end)
    llama31_3b_item = get_explore_llama(activation_model,
                                        attn=False, st=st, end=end)

    atlas = get_atlas(combine_regions=combine_regions)
    ROIs = atlas['ROIs']

    ts = []
    vals_attn_l = []
    vals_item_l = []
    for ROI in ROIs:
        kw_attn = get_base_kw(ROI, model=llama31_3b_attn,
                              fps='all', big_voxelwise=False,
                              local=False)
        process_allow_misses(kw_attn, ROI)
        _, vals_attn = pickle_wrap(run_layer, kwargs=kw_attn,
                                   verbose=-1, easy_override=False,
                                   dir_branches=100)
        vals_attn_l.append(vals_attn)

        kw_item = get_base_kw(ROI, model=llama31_3b_item,
                              fps='all', big_voxelwise=False,
                              local=False)
        process_allow_misses(kw_item, ROI)
        _, vals_item = pickle_wrap(run_layer, kwargs=kw_item,
                                   verbose=-1, easy_override=False,
                                   dir_branches=100)
        vals_item_l.append(vals_item)

    vals_attn_l = np.array(vals_attn_l)
    vals_attn_M = np.nanmean(vals_attn_l, axis=0)
    vals_item_l = np.array(vals_item_l)
    vals_item_M = np.nanmean(vals_item_l, axis=0)
    for i, ROI in enumerate(ROIs):
        vals_attn = vals_attn_l[i]# - vals_attn_M
        vals_item = vals_item_l[i]# - vals_item_M
        t, p = stats.ttest_rel(vals_attn, vals_item, nan_policy='omit') # positive: attn > item
        M_attn = np.nanmean(vals_attn) * 1_000
        M_item = np.nanmean(vals_item) * 1_000
        if np.abs(t) > 2:
            print(f'{ROI}: attn {M_attn:.1f} vs. item {M_item:.1f}, {t=:.3f}')
        ts.append(t)

    vmax = np.quantile(ts, 0.9)
    # vmin = -vmax
    vmin = np.quantile(ts, 0.1)
    vabs = np.quantile(np.abs(ts), 0.9)
    print(f'90th percentile: {vmax:.3f}')
    print(f'10th percentile: {vmin:.3f}')
    title = 'Attn vs. Item'
    title += f'({st} - {end})'
    my_plot_surf(ts, atlas, title, vmax=vabs, thresh=2,
                 only_positive=False,
                 # cmap='warm_cool'
                 )


def run_all_ROIs(combine_regions=False, st=8, end=None, attn=True,
                 activation_model='meta-llama/Llama-3.2-3b',
                 do_M=False, obj_task=True):
    llama31_3b = get_explore_llama(activation_model,
                                   attn=attn, st=st, end=end,
                                   do_M=do_M, last_only=False)
    model = llama31_3b

    # model = get_explore_BERT('simCSE', do_M=False, st=2, end=10)
    atlas = get_atlas(combine_regions=combine_regions)
    ROIs = atlas['ROIs']

    ts = []
    for ROI in ROIs:
        if obj_task:
            fps = 'obj'
        else:
            fps = 'non_obj'
        # fps = ['con7_fMRI']
        kw = get_base_kw(ROI, model=model, fps=fps,
                         big_voxelwise=False, local=False)
        process_allow_misses(kw, ROI)
        t, vals = pickle_wrap(run_layer, kwargs=kw,
                              verbose=-1, easy_override=False,
                              dir_branches=100)
        # if np.abs(t) > 2:
        print(f'{ROI}: {t=:.3f}')
        ts.append(t)

    vmax = np.quantile(np.abs(ts), 0.95)
    title = 'Attn' if attn else 'Item'


    if do_M:
        title = 'Object embedding (scene averages)\n'
    else:
        if fps == 'obj':
            title = 'Object embedding (scene → object)\n'
        else:
            title = 'Object embedding (one scene)\n'

    title += f'(layers {st} - {end})\n'
    if isinstance(fps, list):
        title = str(fps)
        cmap = 'Purples'
    elif fps == 'obj':
        title += 'Encoding task'
        if do_M:
            cmap = 'Reds'
        else:
            cmap = 'Blues'
    else:
        if do_M:
            cmap = 'Reds'
        else:
            cmap = 'Greens'
        title += 'Non-encoding tasks'


    # if cross:
    #     title = 'Object embedding (one context)\n'
    #     title += f'(layers {st} - {end})\n'
    #     title += f'Non-encoding tasks only'
    #     cmap = 'Greens'
    # elif do_M:
    #     title = 'Object embedding (scene averages)\n'
    #     title += f'(layers {st} - {end})\n'
    #     title += f'Non-encoding tasks'
    #     cmap = 'hot'
    #     cmap = 'Reds'
    # else:
    #     title = 'Object embedding (scene → object)\n'
    #     title += f'(layers {st} - {end})\n'
    #     title += f'Encoding task only'
    #     cmap = 'Blues'
    my_plot_surf(ts, atlas, title, vmax=6, thresh=2,
                 only_positive=True, cmap=cmap)


if __name__ == '__main__':
    tick = 4
    ACTIVATION_MODEL = 'meta-llama/Llama-3.2-3b'


    # item_vs_attn_ROIs(combine_regions=False, activation_model=ACTIVATION_MODEL,
    #                   st=24, end=28)
    # quit()
    # run_all_ROIs(attn=True, activation_model=ACTIVATION_MODEL, st=4, end=8)
    # run_all_ROIs(attn=True, activation_model=ACTIVATION_MODEL, st=16, end=24)
    # quit()

    for st in range(4, 20, tick):
        run_all_ROIs(attn=True, activation_model=ACTIVATION_MODEL, st=st, end=st+tick,
                     do_M=False, obj_task=True)
        continue
        # run_all_ROIs(attn=False, activation_model=ACTIVATION_MODEL, st=st, end=st+tick,
        #              do_M=True, obj_task=True)
        # continue

        run_all_ROIs(attn=False, activation_model=ACTIVATION_MODEL, st=st, end=st+tick,
                     do_M=True, obj_task=False)
        quit()
        run_all_ROIs(attn=False, activation_model=ACTIVATION_MODEL, st=st, end=st+tick,
                     do_M=False, obj_task=False)
