
from Utils.atlas_funcs import get_atlas
from collections import Counter
from nilearn import image, plotting
import matplotlib.pyplot as plt
import numpy as np

# suppress
# RuntimeWarning: invalid value encountered in cast
#   BN_data = BN_atlas['maps'].get_fdata().astype(int)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def get_schaef_ROIs(BNA_idx, flip=True):
    search_atlas = get_atlas(combine_regions=True, lifu_labels=True)
    search_data = search_atlas['maps'].get_fdata().astype(int)

    target_atlas = get_atlas(combine_regions=True, schaefer=(True, 400))
    target_data = target_atlas['maps'].get_fdata().astype(int)
    # print('-' * 50)

    if flip:
        search_atlas_ = search_atlas
        search_atlas = target_atlas
        target_atlas = search_atlas_

        search_data_ = search_data
        search_data = target_data
        target_data = search_data_

    # print(f'{BNA_idx=}')
    search_ROI = search_data == BNA_idx
    # # print(np.argwhere(search_ROI))
    # # quit()
    # print(search_atlas["ROIs"][BNA_idx - 1])
    # min_val = np.min(search_data)
    # print(f'{min_val=}')
    # max_val = np.max(search_data)
    # print(f'{max_val=}')
    # search_ROI_img = image.new_img_like(search_atlas['maps'], search_ROI)
    # plotting.plot_roi(search_ROI_img,
    #                   title=f'{BNA_idx} ({search_atlas["ROIs"][BNA_idx - 1]})')
    # plt.show()
    # quit()

    sch_ROI_vals = target_data[search_ROI]
    sch_labels = target_atlas['ROIs']
    cnt0 = sch_ROI_vals == 0
    sch_ROI_labels = [sch_labels[i - 1] for i in sch_ROI_vals if i != 0]
    sch_ROI_labels += ['None'] * sum(cnt0)
    # print(f'{BNA_idx=}, {sch_ROI_vals=}')
    sch_ROI_labels = Counter(sch_ROI_labels)
    BN_label = search_atlas['ROIs'][BNA_idx - 1]
    print(f'{BNA_idx} ({BN_label}): {sch_ROI_labels}')



def do_all_ROI_ROI():
    for i in range(1_000):
        # i = 272
        try:
            get_schaef_ROIs(i + 1)
        except IndexError:
            break
        # quit()

if __name__ == '__main__':
    do_all_ROI_ROI()
