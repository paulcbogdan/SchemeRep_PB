from Utils.pickle_wrap_funcs import pickle_wrap
from connRSA_finalizing.plot_bars_explore import plot_Fig5_v2_kw

def get_target_roi2baddies():
    target_roi2baddies = {'Parietal': [('llama', 'gate_proj_in', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'gate_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'down_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'act_fn_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'q_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'k_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'v_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_output', 0, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_output', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'input', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'down_proj_in', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'q_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'k_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 1, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'input', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'down_proj_in', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'q_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'k_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 2, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'gate_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'down_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_output', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_output', 12, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_output', 12, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'attn_weights', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                       ('llama', 'gate_proj_out', 14, 'scn', 'meta-llama/Llama-3.2-3b', True)],
                          'PFC': [('llama', 'gate_proj_in', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'act_fn_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'q_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'v_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'input', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_in', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_in', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_out', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_out', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'act_fn_out', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'q_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'v_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 1, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'input', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_in', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_in', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_out', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'up_proj_out', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_out', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'act_fn_out', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'q_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'v_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 2, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'input', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_in', 3, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'q_proj', 3, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 3, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 3, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'input', 3, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 5, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_in', 7, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 8, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'down_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 10, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 12, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 12, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_out', 14, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 14, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 14, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 15, 'obj', 'meta-llama/Llama-3.2-3b', True)],
                          'Occipital': [('llama', 'gate_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'down_proj_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'act_fn_out', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'q_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'k_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'q_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'k_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 1, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'q_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'k_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'k_proj', 4, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 7, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 8, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 8, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'gate_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 9, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 10, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 12, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 12, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 14, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 14, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_weights', 15, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                        ('llama', 'attn_output', 15, 'obj', 'meta-llama/Llama-3.2-3b', True)],
                          'ITL': [('llama', 'q_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 0, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 0, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'q_proj', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 1, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 1, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 1, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 2, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 2, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 2, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 3, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 3, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 4, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 4, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 5, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 5, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 6, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 7, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 8, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 8, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'gate_proj_in', 9, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 9, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 10, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 11, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 12, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 13, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 14, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 14, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'k_proj', 15, 'scn', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_weights', 15, 'obj', 'meta-llama/Llama-3.2-3b', True),
                                  ('llama', 'attn_output', 15, 'obj', 'meta-llama/Llama-3.2-3b', True)]}
    return target_roi2baddies

def get_semantic_l(target_ROI):
    activation_model = 'meta-llama/Llama-3.2-3b'
    # activation_model = 'meta-llama/Llama-3.1-70b'
    normalize = True
    all_llama_layers = list(range(16, 28))
    all_llama_cats = ['gate_proj_in', 'down_proj_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']
    all_llama_cats = ['gate_proj_in', 'down_proj_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', #'attn_weights',
                      'attn_output',
                      'input']
    # all_llama_cats = ['gate_proj_in']

    target_roi2baddies = get_target_roi2baddies()
    semantic_l = []
    for llama_layer in all_llama_layers:
        for llama_cat in all_llama_cats:
            if llama_cat in ['attn_weights', 'attn_output']:
                continue
                semantic = ('llama', llama_cat, llama_layer, 'obj', activation_model,
                            normalize)
                if semantic in target_roi2baddies[target_ROI]:
                    continue
                semantic_l.append(semantic)
            semantic = ('llama', llama_cat, llama_layer, 'scn', activation_model,
                        normalize)
            if semantic in target_roi2baddies[target_ROI]:
                continue
            semantic_l.append(semantic)
    print(semantic_l)
    # quit()
    return semantic_l

def test_llama_combined(big_voxelwise=True):
    target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']

    fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI',  'vis7_fMRI' ]
    fps = ['obj7_fMRI']


    for i, target_ROI in enumerate(target_ROIs):
        semantic_l = get_semantic_l(target_ROI)


        kwargs = {'semantic': semantic_l,
                  'fp': None,
                  'trial_similarity': 'corr',
                  'second_order': 'spear',
                  'RDM_method': 'within_nan',
                  'stdize_by_run': False,
                  'regress_row': False,
                  }
        (t_corr, t_local_corr_1samp, t_dist_corr_1samp,
         t_beta, t_local_beta_1samp, t_dist_beta_1samp) = (
            pickle_wrap(plot_Fig5_v2_kw, kwargs={'kw': kwargs, 'fps': fps,
                                                 'big_voxelwise': big_voxelwise,
                                                 'region': target_ROI,
                                                 'std': False,
                                                 'easy_override': False,
                                                 'plot_i': i},
                        verbose=-1))


        print(f'\n-+- {target_ROI} -+-')
        print(f'Local vs. distributed, corr, t = {t_corr:.2f}')
        print(f'\tLocal 1-samp, corr: t = {t_local_corr_1samp:.2f}')
        print(f'\tDist 1-samp, corr: t = {t_dist_corr_1samp:.2f}')
        print(f'Local vs. distributed, beta, t = {t_beta:.2f}')
        print(f'\tLocal 1-samp, beta: t = {t_local_beta_1samp:.2f}')
        print(f'\tDist 1-samp, beta: t = {t_dist_beta_1samp:.2f}')

if __name__ == '__main__':
    test_llama_combined()









