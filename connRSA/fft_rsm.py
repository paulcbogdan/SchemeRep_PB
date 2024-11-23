import numpy as np
from scipy import stats as stats

import utils
from connRSA.fft_funcs import get_uniform_size_ffts, get_ffts, ffts_reconstruct, extract_fz
from connRSA.make_RSM_stim import get_sn_fp_stim_RSM


def get_fz_neural_RSM(sn, region, fp, fz, uniform_size, eight_corners, do_mag,
                      do_reconstruct):

    if '_L' not in region and '_R' not in region:
        neural_RSM_L = utils.pickle_wrap(get_fz_neural_RSM, None,
                                         kwargs={'sn': sn, 'region': f'{region}_L',
                                                 'fp': fp, 'fz': fz,
                                                 'uniform_size': uniform_size,
                                                 'eight_corners': eight_corners,
                                                 'do_mag': do_mag,
                                                 'do_reconstruct': do_reconstruct},
                                         verbose=-1, easy_override=False,
                                         dir_branches=100)
        neural_RSM_R = utils.pickle_wrap(get_fz_neural_RSM, None,
                                         kwargs={'sn': sn, 'region': f'{region}_R',
                                                 'fp': fp, 'fz': fz,
                                                 'uniform_size': uniform_size,
                                                 'eight_corners': eight_corners,
                                                 'do_mag': do_mag,
                                                 'do_reconstruct': do_reconstruct},
                                         verbose=-1, easy_override=False,
                                         dir_branches=100)
        neural_RSM = (neural_RSM_L + neural_RSM_R) / 2
        return neural_RSM
    if fz == 'sanity':
        pass

    elif uniform_size:
        ffts, mask, orig_size = utils.pickle_wrap(get_uniform_size_ffts, None,
                                                  kwargs={'sn': sn, 'region': region,
                                                          'fp': fp, },
                                                  verbose=-1, easy_override=False,
                                                  )

        if ffts.shape[0] > 49:
            print(f'Redoing: {ffts.shape=}')
            ffts, mask, orig_size = utils.pickle_wrap(get_uniform_size_ffts, None,
                                                      kwargs={'sn': sn, 'region': region,
                                                              'fp': fp, },
                                                      verbose=-1, easy_override=True,
                                                      )
    else:
        ffts, mask = utils.pickle_wrap(get_ffts, None,
                                       kwargs={'sn': sn, 'region': region,
                                               'fp': fp,},
                                       verbose=-1, easy_override=False)
        orig_size = ffts.shape

    if do_mag:
        ffts = np.abs(ffts)


    if do_reconstruct:
        # fp_step2 = fr'cache/fft_reconstruct/{sn}_{region}_{fp}_{fz}_{uniform_size}.pkl'
        # # eight_corners = False
        # kw = {'ffts': ffts, 'mask': mask, 'fz': fz,
        #       'eight_corners': eight_corners}
        # if uniform_size:
        #     kw['resize'] = orig_size
        # vals = utils.pickle_wrap(ffts_reconstruct, fp_step2,
        #                          kwargs=kw,
        #                          verbose=-1, easy_override=False)
        vals = ffts_reconstruct(ffts, mask, fz, resize=orig_size,
                                eight_corners=eight_corners)

        # TODO: ADD     img_box_g = stats.zscore(img_box_g, axis=-1) # NEEDED FOR PERFECT SIMILARITY?

        neural_RSM = np.corrcoef(vals)
    else:
        vals = extract_fz(ffts, fz, eight_corners=eight_corners)
        neural_RSM = np.corrcoef(vals)
        neural_RSM = np.real(neural_RSM)
    return neural_RSM


def run_FFT_RSA2(sn, region, fp, fz=(1, 5), semantic=True, layer=None,
                 do_mag=False, do_reconstruct=False, uniform_size=True,
                 eight_corners=False, ):
    assert not (do_mag and do_reconstruct)
    # print('-')

    neural_RSM = utils.pickle_wrap(get_fz_neural_RSM, None,
                                   kwargs={'sn': sn, 'region': region,
                                           'fp': fp, 'fz': fz,
                                           'uniform_size': uniform_size,
                                           'eight_corners': eight_corners,
                                           'do_mag': do_mag,
                                           'do_reconstruct': do_reconstruct},
                                   verbose=-1, easy_override=False,
                                   dir_branches=100)


    neural_RSM[np.diag_indices_from(neural_RSM)] = np.nan
    trils = np.tril_indices_from(neural_RSM, k=-1)
    neural_RSM = neural_RSM[trils]

    stim_RSM = get_sn_fp_stim_RSM(sn, fp, semantic=semantic,
                                  layer=layer)
    stim_RSM = stim_RSM[trils]
    r, _ = stats.spearmanr(neural_RSM, stim_RSM, nan_policy='omit')
    return r
