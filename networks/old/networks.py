def prep_networks(network_setting=1):
    if not network_setting:
        return None
    if network_setting == 1:
        networks = {
            'Occipital': ['EVC', 'LOC', 'sOcG'],
            'Ventral': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG'],
            'Dorsal': ['SPL', 'IPL', 'Pcun', 'pSTS'],
            #'dPFC': ['IFG', 'MFG', 'SFG'],
            #'PFC_Occ': ['IFG', 'MFG', 'SFG', 'EVC', 'LOC', 'sOcG'],
            #'FPCN': ['IFG', 'MFG', 'SFG', 'SPL', 'IPL', 'pSTS']
        }
    elif network_setting == 2:
        networks = {
            # 'Frontal': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', ],
            # 'PFC_sub': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'Amyg', 'Hipp',
            #             'Str', 'Tha'],
            'else': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS', 'SPL',
                     'IPL', 'Pcun', 'PoG', 'INS', 'CG', 'Amyg', 'Hipp', 'Str',
                     'Tha', 'ACC', 'PCC'],
            'else_cortical': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL', 'pSTS',
                              'SPL', 'IPL', 'Pcun', 'PoG', 'INS', 'CG',
                              'ACC', 'PCC'],
            'sub': ['Amyg', 'Hipp', 'Str', 'Tha'],
        }
    # elif setting == 3:
    #     networks = {
    #         'Sanity': (['IFG', 'MFG', 'INS']),
    #         'PFC_Occ': (['IFG', 'MFG', 'SFG', 'OrG', 'EVC', 'LOC']),
    #         'Hipp_Occ': (['Hipp', 'EVC', 'LOC']),
    #         'Parietal_Occ': (['SPL', 'IPL', 'pSTS', 'Pcun', 'EVC', 'LOC']),
    #         'Ventral_Occ': (['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'EVC', 'LOC']),
    #     }
    elif network_setting == 3:
        networks = {
            'Sanity': (['IFG', 'MFG'], ['INS']),
            'PFC_Occ': (['IFG', 'MFG', 'SFG', 'OrG'], ['EVC', 'LOC']),
            'Hipp_Occ': (['Hipp'], ['EVC', 'LOC']),
            'Parietal_Occ': (['SPL', 'IPL', 'pSTS', 'Pcun'], ['EVC', 'LOC']),
            'Ventral_Occ': (['ITG', 'FuG', 'PhG', 'ATL', 'MTG'], ['EVC', 'LOC']),
        }
    elif network_setting == 4:
        networks = {
            'PFC': ['SFG', 'MFG', 'IFG', 'OrG',],
        }
    elif network_setting == 5:
        raise ValueError('setting 5 is bad don\'t use it')
        # networks = {
        #     'Frontal_CG': ['SFG', 'MFG', 'IFG', 'OrG'],
        # }
    elif network_setting == 6:
        networks = {
            'Frontal_CG': ['SFG', 'MFG', 'IFG', 'OrG'],
            # 'dPFC_Occ': (['IFG', 'MFG', 'SFG'], ['EVC', 'LOC']),
            # 'dlPFC_Occ': (['IFG', 'MFG'], ['EVC', 'LOC']),
            'PFC_Hipp': ['IFG', 'MFG', 'SFG', 'Hipp', 'OrG'],
            'dPFC': ['IFG', 'MFG', 'SFG'],
            'DMN': ['OrG', 'CG', 'Pcun', 'IPL'],
            'Salience': ['INS', 'CG'],
            'FPCN': ['MFG', 'IFG', 'IPL'],
            'FPCN_CG': ['MFG', 'IFG', 'IPL', 'CG'],
            'dlPFC': ['IFG', 'MFG'],
            'mPFC_hipp': ['OrG', 'Hipp'],
        }
    elif network_setting == 7:
        networks = {'else_ventral': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL',
                                     'pSTS', 'SPL', 'IPL', 'Pcun', 'PoG', 'INS',
                                     'CG', 'Amyg', 'Hipp', 'Str', 'Tha', 'ITG',
                                     'FuG', 'PhG', 'ATL', 'MTG']}
    elif network_setting == 8:
        networks = {'ventral_hipp': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'Hipp'],
                    'temporal': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'STG', 'pSTS'],
                    'dorsal_proper': ['SPL', 'IPL', 'Pcun', 'PoG'],
                    }
    elif network_setting == 9:
        networks = {'MTL': ['PhG', 'ATL', 'Hipp']} # 'ITG',
    elif network_setting == 10:
        networks = {'whole_brain': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL',
                                     'pSTS', 'SPL', 'IPL', 'Pcun', 'PoG', 'INS',
                                     'CG', 'Amyg', 'Hipp', 'Str', 'Tha', 'ITG',
                                     'FuG', 'PhG', 'ATL', 'MTG',
                                     'EVC', 'LOC', 'sOcG'],
                    'perceptual': ['ITG', 'FuG', 'PhG', 'ATL', 'MTG', 'STG', 'pSTS',
                                   'SPL', 'IPL', 'Pcun', 'PoG',
                                   'EVC', 'LOC', 'sOcG']}
    elif network_setting == 11:
        networks = {'full_frontal': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG', 'PCL'],
                    'full_frontal_CG': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG',
                                        'PCL', 'CG'],}
    elif network_setting == 12:
        networks = {'dPFC': ['SFG', 'MFG', 'IFG',
                             'PFCl', 'PFCd'],
                    'PFC': ['SFG', 'MFG', 'IFG', 'OrG',
                            'PFCd', 'PFCl', 'PFCmp', 'PFClv', 'PFCm', 'PFCv',
                            'OFC']}
    elif network_setting == 13:
        networks = {'full_frontal': ['SFG', 'MFG', 'IFG', 'OrG', 'PrG'],
                    'dorsal_frontal': ['SFG', 'MFG', 'IFG', 'PrG'],
                    'prefrontal': ['SFG', 'MFG', 'IFG', 'OrG']}
    elif network_setting == 14:
        networks = {'PFC': ['SFG', 'MFG', 'IFG', 'OrG'],
                    'PFC_ACC': ['SFG', 'MFG', 'IFG', 'OrG', 'ACC'],
                    'FP': ['MFG', 'IFG', 'IPL', 'SPL']}
    elif network_setting == 15:
        networks = {'PFC_lTemp': ['SFG', 'MFG', 'IFG', 'OrG',
                                              'ATL', 'STG', 'MTG']}
    elif network_setting == 16:
        networks = {'PFC': ['SFG', 'MFG', 'IFG', 'OrG']}
    elif network_setting == 17:
        networks = {'perceptual': ['EVC', 'LOC', 'sOcG',
                                   'ITG', 'FuG', 'PhG', 'ATL', 'MTG',
                                   'SPL', 'IPL', 'Pcun', 'pSTS']}
    elif network_setting == -1:
        networks = {}
    # elif setting == 7:
    #     networks = {
    #         'dPFC_Occ': (['IFG', 'MFG', 'SFG'], ['EVC', 'LOC']),
    #         'dlPFC_Occ': (['IFG', 'MFG'], ['EVC', 'LOC']),
    #     }
    else:
        raise ValueError(f'Unknown setting: {network_setting}')
    return networks
