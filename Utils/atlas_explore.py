from matplotlib import pyplot as plt
from nilearn import image, plotting

from Utils.atlas_funcs import get_atlas
import numpy as np


def plot_ROI(schaefer=True, ROI='IPL_L', combine_regions=True):
    if schaefer:
        schaefer = (schaefer, 400)
    else:
        pass
    atlas = get_atlas(combine_regions=combine_regions,
                      schaefer=schaefer)
    # if isinstance(ROI, str):
    #     idx = atlas['labels'].index(ROI)
    # if not idx:
    #     raise ValueError(f'ROI {ROI} not found in atlas: {atlas["labels"]=}')
    idx = 47
    # print(idx)
    # quit()
    data = image.load_img(atlas['maps']).get_fdata()
    ROI_data = data == (idx + 1)
    # print(data == 121)

    # print(np.sum(ROI_data))
    # quit()
    plotting.plot_roi(image.new_img_like(atlas['maps'], ROI_data),
                      title=f'{ROI}')
    plt.show()

if __name__ == '__main__':
    plot_ROI()
    # quit()