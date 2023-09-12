from nilearn.plotting import plot_roi, view_img
from nilearn import image
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_BN_atlas():
    img = image.load_img(r'cache/BN_Atlas_246_2mm.nii.gz')
    labels = pd.read_csv(r'cache/BNA_labels.txt', header=None)[0].to_list()
    print(f'{labels=}')
    atlas = {'maps': img, 'labels': labels}
    return atlas

if __name__ == '__main__':
    get_BN_atlas()