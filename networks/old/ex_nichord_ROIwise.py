import os

from atlas_utils import get_atlas
import numpy as np

from old.modularity import plot_nichord

if __name__ == '__main__':
    atlas = get_atlas()
    coords = atlas['coords']
    dir_out = 'result_pics/FC'
    fn = 'ex_ROIwise.png'
    edges = [(20, i) for i in range(0, len(coords))]
    edge_weights = np.random.normal(size=len(edges))
    plot_nichord(coords, fn, 'Example ROI-wise classification', dir_out,
                 edges=edges, edge_weights=edge_weights)


