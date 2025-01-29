import numpy as np

from connRSA.plot_bars_explore import get_explore_llama
from llama.get_obj_scn_vecs import get_sn_fp_llama_RSM_l
import scipy.stats as stats
import matplotlib.pyplot as plt

def attn_x_item_corr():
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'
    activation_model = 'meta-llama/Llama-3.2-3b'
    model_attn = get_explore_llama(activation_model, attn=True,
                                   normalize=1)
    model_item = get_explore_llama(activation_model, attn=False,
                                   normalize=1)

    rs = []
    for sn in [102, 103, 104]:
        RSM_attn = get_sn_fp_llama_RSM_l(sn, 'obj7_fMRI', model_attn)
        plt.imshow(RSM_attn)
        plt.show()
        RSM_item = get_sn_fp_llama_RSM_l(sn, 'obj7_fMRI', model_item)
        trils = np.tril_indices_from(RSM_attn, k=-1)
        RSM_attn = RSM_attn[trils]
        RSM_item = RSM_item[trils]
        r, p = stats.spearmanr(RSM_attn, RSM_item, nan_policy='omit')
        rs.append(r)
        print(f'{sn} | {r=:.3f}')
    M_r = np.mean(rs)
    print(f'attn x item M = {M_r:.2f}')


if __name__ == '__main__':
    attn_x_item_corr()