import numpy as np



def get_FC_between_ROIs(conn_trials, p_mod0, p_mod1, trialwise=True,
                        transpose=True):
    if transpose:
        conn_trials = np.transpose(conn_trials, (0, 1, 4, 2, 3))
    # conn_trials_cross = get_partition_cross(conn_trials, p_mod0, p_mod1)

    meshy = np.ix_(p_mod0, p_mod1)
    slicer = tuple([slice(None)] * (conn_trials.ndim - 2) + [meshy[0], meshy[1]])
    conn_trials_cross = conn_trials[slicer]

    flat_cross = np.reshape(conn_trials_cross, (conn_trials_cross.shape[0],
                                                conn_trials_cross.shape[1],
                                                conn_trials_cross.shape[2], -1))
    if trialwise:
        agg_zs = np.nanmean(flat_cross, axis=-1)
        agg_zs = np.nanmean(agg_zs, axis=1) # omit inc axis
        return agg_zs
    else:
        rs = np.nanmean(flat_cross, axis=2)
        agg_rs = np.nanmean(rs, axis=-1)
        return agg_rs
