
def make_stim(feat_classes=2, feat_categories=(1, 2), feat_per_category=(10, 5)):
    feats = []
    for c in range(feat_classes):
        feats.append(np.random.normal(size=(feat_categories[c],
                                            feat_per_category[0])))
    return feats








