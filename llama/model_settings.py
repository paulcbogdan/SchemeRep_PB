from functools import cache


@cache
def get_explore_llama(activation_model='meta-llama/Llama-3.3-70b-Instruct',
                      attn=False, normalize=True, st=0, end=None,
                      do_prod=False, do_M=False, last_only=False,
                      include_scn=False
                      ):
    # Redundant: gate_proj_in & up_proj_in
    # Redundant: act_fn_in & gate_proj_out
    all_llama_cats = ['gate_proj_in', 'up_proj_in', 'down_proj_in', 'act_fn_in',
                      'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
                      'q_proj', 'k_proj', 'v_proj', 'attn_weights', 'attn_output',
                      'input']
    if not isinstance(attn, bool):
        all_llama_cats = [attn]
    elif attn:
        all_llama_cats = ['attn_weights']
    else:
        all_llama_cats = ['input']
    # if isinstance(attn, bool) and attn:
    #     assert not do_M
    # print(activation_model)
    # if activation_model == 'meta-llama/Llama-3.2-3b':
    if ('-1b' in activation_model or
            (isinstance(activation_model, tuple) and '-1b' in activation_model[0])):
        all_llama_layers = list(range(st, 16 if end is None else end))
    elif ('-3b' in activation_model or
            (isinstance(activation_model, tuple) and '-3b' in activation_model[0])):
        all_llama_layers = list(range(st, 28 if end is None else end))
    elif ('-7b' in activation_model or
          (isinstance(activation_model, tuple) and '-7b' in activation_model[0])):
        all_llama_layers = list(range(st, 32 if end is None else end))
    else:
        all_llama_layers = list(range(st, 80 if end is None else end))

    # all_llama_layers = list(range(0, 10))

    semantic_l = []
    for llama_layer in all_llama_layers:
        if last_only:
            llama_layer = (llama_layer, True)
        for llama_cat in all_llama_cats:
            if isinstance(do_M, str):
                semantic_l.append(('llama', llama_cat, llama_layer, do_M, activation_model,
                                   normalize))
            elif do_M:
                if include_scn:
                    semantic_l.append(('llama', llama_cat, llama_layer, 'scn_M', activation_model,
                                       normalize))
                semantic_l.append(('llama', llama_cat, llama_layer, 'obj_M', activation_model,
                                   normalize))
            elif do_prod:
                semantic_l.append(('llama', llama_cat, llama_layer, 'prod', activation_model,
                                   normalize))
            else:
                # if llama_cat in ['attn_weights', 'attn_output']:
                if include_scn:
                    semantic_l.append(('llama', llama_cat, llama_layer, 'scn', activation_model,
                                       normalize))
                semantic_l.append(('llama', llama_cat, llama_layer, 'obj', activation_model,
                                   normalize))

    return semantic_l


def get_explore_BERT(bert_type='BERT', st=0, end=13, do_M=False):
    layers = list(range(st, end))
    semantic_l = []
    normalize = True
    for layer in layers:
        obj_str = 'obj_M' if do_M else 'obj'
        semantic = ('BERT', bert_type, layer, obj_str, None, normalize)
        semantic_l.append(semantic)
    return semantic_l


def flat_dict_to_nested(flat_dict):
    nested_dict = {}
    for (a, b, c), value in flat_dict.items():
        # Create nested dictionaries if they don't exist
        if a not in nested_dict:
            nested_dict[a] = {}

        if b not in nested_dict[a]:
            nested_dict[a][b] = {}

        nested_dict[a][b][c] = value
    return nested_dict


def get_base_kw(region, model, fps='obj', big_voxelwise=True, local=False):
    kw_inner = {'trial_similarity': 'corr',
                'second_order': 'spear',
                'RDM_method': 'within_nan',
                'stdize_by_run': False,
                'regress_row': False,
                'semantic': model,
                }
    if isinstance(fps, str):
        if fps.lower() == 'non_obj':
            fps = ['bl7_fMRI', 'con7_fMRI', 'vis7_fMRI']
        elif fps.lower() == 'obj':
            fps = ['obj7_fMRI']
        elif fps.lower() == 'all':
            fps = ['bl7_fMRI', 'obj7_fMRI', 'con7_fMRI', 'vis7_fMRI']
        else:
            raise ValueError

    kwargs = {'kw': kw_inner, 'fps': fps, 'big_voxelwise': big_voxelwise,
              'region': region, 'easy_override': False, 'local': local}
    return kwargs
