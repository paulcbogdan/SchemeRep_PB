from modularity import get_main_partitions, get_BNA_coords
from networks.old.network_funcs import load_FC_for_Lifu
from utils import pickle_wrap

if __name__ == '__main__':
    fp = 'obj3_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3)}

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(None,
                                                                  load_FC_for_Lifu, kwargs=kwargs,
                                                                  verbose=1, easy_override=False,
                                                                  cache_dir='../cache')

    atlas_name = kwargs['atlas_name'] if 'atlas_name' in kwargs else 'BNA'
    coords = get_BNA_coords(atlas_name)
    get_main_partitions(
            sn_conn, plot=True, threshold=0.95, coords=coords,
            overlapping='angel')
