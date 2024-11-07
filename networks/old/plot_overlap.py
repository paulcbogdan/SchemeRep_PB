from Study1A.modularity_funcs import get_main_partitions, get_BNA_coords
from Study1A.load_data_Study1A import load_FC
from utils import pickle_wrap

if __name__ == '__main__':
    fp = 'obj3_fMRI'
    kwargs = {'fp': fp, 'split': False,
              'key': 'inc',
              'key_vals': (1, 3)}

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = pickle_wrap(load_FC, None, kwargs=kwargs,
                                                                  easy_override=False, verbose=1, cache_dir='../cache')

    atlas_name = kwargs['atlas_name'] if 'atlas_name' in kwargs else 'BNA'
    coords = get_BNA_coords(atlas_name)
    get_main_partitions(
            sn_conn, plot=True, threshold=0.95, coords=coords,
            overlapping='angel')
