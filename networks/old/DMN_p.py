
def get_DMN_p(exclude_ps=None):
    atlas = get_atlas()

    from nichord.coord_labeler import get_idx_to_label
    idx_to_label = pickle_wrap(get_idx_to_label, None, kwargs={'coords': atlas['coords'],
                                                               'atlas': 'yeo'})
    DMN_idxs = [idx for idx, label in idx_to_label.items() if 'DMN' in label]
    # print(DMN_idxs)


    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=False, scrub=False)
    # exclude_ps = p_dorsal + p_ventral

    for i in DMN_idxs:
        BNA_label = atlas['ROIs'][i]
        if i in p_d_pos:
            print(f'{BNA_label} ({i}): In: Dor-Pos')
        elif i in p_d_ant:
            print(f'{BNA_label} ({i}): In: Dor-Ant')
        elif i in p_v_pos:
            print(f'{BNA_label} ({i}): In: Ven-Pos')
        elif i in p_v_ant:
            print(f'{BNA_label} ({i}): In: Ven-Ant')
        else:
            print(f'{BNA_label} ({i}): Not in any')

    print('--------')

    cnt = defaultdict(lambda: 0)
    for i in p_d_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_d_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Dor-Ant: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_pos:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Pos: {cnt}')

    cnt = defaultdict(lambda: 0)
    for i in p_v_ant:
        yeo = idx_to_label[i]
        cnt[yeo] += 1
    cnt = dict(cnt)
    print(f'Ven-Ant: {cnt}')

