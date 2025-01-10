from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
import numpy as np

from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from networks.old.networks import prep_networks


def get_ROI_size(idx):
    atlas = get_atlas(combine_regions=False)
    mp = atlas['maps'].get_fdata()
    ROI = mp == (idx + 1)
    ROI_size = np.sum(ROI)
    return ROI_size


def get_ROI_sizes(target_ROI='ITL'):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[target_ROI])[target_ROI])
    ROIs_match = [i for region in regions
                  for i, ROI in enumerate(get_BNA_ROIs()) if region in ROI]
    sizes = [get_ROI_size(ROI) for ROI in ROIs_match]
    M_voxels = np.mean(sizes)
    M_mm3 = M_voxels * 8
    num_voxels = sum(sizes)
    mm3 = num_voxels * 8
    print(f'{target_ROI} | {num_voxels} voxels | {mm3:,} mm3')
    print(f'\tMean: {M_voxels:.1f} voxels | {M_mm3:.1f} mm3')
    return sizes

if __name__ == '__main__':
    areas = ['Occipital', 'ITL', 'PFC', 'Parietal', 'cortical']
    sizes_all = []
    for area in areas:
        sizes_all.extend(get_ROI_sizes(area))
    M_all = np.mean(sizes_all)
    M_all_mm3 = M_all * 8
    # print(f'All areas | {M_all:.1f} voxels, {M_all_mm3:.1f} mm3')
