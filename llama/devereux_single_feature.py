from llama.Rissman_similarity_analysis import fit_regularized_models
from llama.devereux_feature_regression import get_final_feats, get_vecs_for_regr
from llama.devereux_llama import get_standard_items_list
from llama.devereux_neuron import get_binary_feat_matrix
import matplotlib.pyplot as plt

from marinate.pkld import pkld

from tqdm import tqdm

@pkld
def regression_one_feature(feature='is_small',
                           pf_thresh=600, cat='gate_proj_in', layer_name=8,
                           activation_model='meta-llama/Llama-3.2-3b',
                           normalize=False, quick=1, item_standard='deve',
                           position=0, symmetric=False,
                           req=5, normalize_regr=False,
                           do_r2=True
                           ):

    vecs = get_vecs_for_regr(activation_model, pf_thresh, item_standard, cat,
                             layer_name, normalize, quick, position, symmetric,
                             store_='both'
                             )

    items, df = get_standard_items_list(pf_thresh, item_standard)

    feat2onehot = get_binary_feat_matrix(items, item_standard, threshold=20,
                                         pf_thresh=pf_thresh, req=req)
    onehot = feat2onehot[feature]
    res = fit_regularized_models(vecs, onehot, normalize=normalize_regr,
                                 do_r2=do_r2)
    return res['Ridge']['r2_score']

def plot_all_feats(cat='gate_proj_in'):
    feats = get_final_feats()[::-1]
    for feat in feats[:5]:
        r2s = []
        for layer_name in tqdm(range(28)):
            r2 = regression_one_feature(feat, cat=cat,
                                        layer_name=layer_name)
            r2s.append(r2)
        plt.plot(r2s, label=feat, alpha=0.5,
                 marker='o', markersize=3)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_all_feats()
