import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
from time import time

from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve
from llama.get_obj_scn_vecs import process_cat_cat_inner


def get_rissman_df(condition=''):
    fp = r'C:\PycharmProjects\SchemeRep\llama\features\WelshRissman_NatCom_all_data.csv'
    df = pd.read_csv(fp)
    df = df[~pd.isna(df['relatedness_judgement'])]
    df = df[df['relatedness_judgement'].isin({'1', '2', '3', '4'})]
    df['cue'] = df['cue'].str.lower()
    df['target'] = df['target'].str.lower()
    df['relatedness_judgment'] = df['relatedness_judgement'].astype(int)
    df_grp = df.groupby(['cue', 'target'])['relatedness_judgment'].mean().reset_index()
    pairs = df_grp[['cue', 'target']].values
    relatedness = df_grp['relatedness_judgment'].values
    return pairs, relatedness, df_grp

def get_rissman_similarity(pair, flip=False,
                           activation_model='meta-llama/Llama-3.2-3b',
                           cat='attn_weights', get_last=False,
                           layer_name=1):
    cat_, inner = process_cat_cat_inner(cat)

    if flip:
        item0 = pair[1].lower()
        item1 = pair[0].lower()
    else:
        item0 = pair[0].lower()
        item1 = pair[1].lower()

    res, fp = pickle_wrap(get_llama_activations_deve,
                          kwargs={'item0': item0, 'item1': item1,
                                  'activation_model': activation_model},
                          easy_override=False, verbose=-1, dir_branches=100,
                          RAM_cache=False, get_fp=True)

    if 'mlp_out' in res:
        del res['mlp_out']
        del res['mlp_in']['down_proj']
        del res['mlp_in']['up_proj']
        del res['mlp_in']['act_fn']
        del res['attn']['q_proj']
        del res['attn']['k_proj']

        with open(fp, 'wb') as f:
            pickle.dump(res, f)


    d_vecs = {}
    for idx_target in [0, 1]:
        if activation_model in ['BERT', 'simCSE']:
            v = res[idx_target][layer_name]
        elif cat in ['attn_weights']:
            if get_last:
                v = res['attn']['attn_weights'][layer_name][idx_target][-1][-1]
            else:
                v = np.nanmean(res['attn']['attn_weights'][layer_name][idx_target],
                               axis=(0, 1))
        else:
            if get_last:
                v = res[inner][cat_][layer_name][idx_target][-1]
            else:
                v = np.nanmean(res[inner][cat_][layer_name][idx_target], axis=0)
            if len(v.shape) > 1:
                v = v.reshape(-1)

        if idx_target == 0:
            if (item0, item1, item0) in d_vecs:
                raise ValueError
            d_vecs[(item0, item1, item0)] = v
        else:
            if (item0, item1, item1) in d_vecs:
                raise ValueError
            d_vecs[(item0, item1, item1)] = v

    return d_vecs

def get_rissman_w2v_d_vecs():
    print('Cooking rissman word2vecs...')
    pairs, _, _ = get_rissman_df()
    from gensim import downloader
    w2vectors = downloader.load('word2vec-google-news-300')
    d_vecs = {}
    for pair in tqdm(pairs, desc='Going through rissman word2vec words'):
        for item in pair:
            try:
                vec = w2vectors[item]
                d_vecs[item] = vec
            except KeyError:
                print(f'Bad {item}')
                raise ValueError
    return d_vecs

def get_w2v_similarity(do_print=False):
    d_vecs = pickle_wrap(get_rissman_w2v_d_vecs, verbose=-1)
    pairs, relatedness, _ = get_rissman_df()
    w2v_similarities = []
    for pair in pairs:
        vec_0 = d_vecs[pair[0]]
        vec_1 = d_vecs[pair[1]]
        r, p = stats.spearmanr(vec_0, vec_1)
        if do_print: print(f'w2v similarity | {pair=}: {r=:.2f}, {p=:.3f}')
        w2v_similarities.append(r)
    r, p = stats.spearmanr(w2v_similarities, relatedness)
    plt.rcParams.update({'font.size': 16})
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.scatter(w2v_similarities, relatedness)
    plt.title(f'Relatedness x word2vec similarity: {r=:.2f}')
    plt.xlabel('Word2vec pair similarity \n(Spearman)')
    plt.ylabel('Avg. reported relatedness')
    plt.tight_layout()
    plt.show()
    print(f'word2vec | relatedness x corr: {r=:.2f}, {p=:.3f}')
    return np.array(w2v_similarities)

def test_w2v_regression():
    d_vecs = pickle_wrap(get_rissman_w2v_d_vecs, verbose=-1)
    pairs, relatedness, _ = get_rissman_df()
    # np.random.shuffle(relatedness)
    prods = []
    for pair in pairs:
        vec_0 = d_vecs[pair[0]]
        vec_1 = d_vecs[pair[1]]
        assert np.isnan(vec_0).sum() == 0
        assert np.isnan(vec_1).sum() == 0
        vec_0 = stats.zscore(vec_0)
        vec_1 = stats.zscore(vec_1)
        prod = vec_0 * vec_1
        prods.append(prod)
    prods = np.array(prods)
    Ms = np.nanmean(prods, axis=1)
    r, p = stats.spearmanr(Ms, relatedness)
    print(f'Plain w2v correlation: {r=:.3f}')
    prods = np.nanmean(prods, axis=1, keepdims=True)
    fit_regularized_models(prods, relatedness)

def fit_regularized_models(X, y, cv_folds=5, random_state=42):

    """
    Fit Lasso, Ridge, and ElasticNet models with cross-validation

    Parameters:
    X: Features matrix (60 x 24)
    y: Target variable
    cv_folds: Number of cross-validation folds
    random_state: Random seed for reproducibility

    Returns:
    Dictionary containing models and their R² scores
    """

    import numpy as np
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_scaled = X

    # Create cross-validation object
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # cv = LeaveOneOut()

    # Initialize models
    lasso = LassoCV(cv=cv, random_state=random_state)
    ridge = RidgeCV(cv=cv)
    elastic = ElasticNetCV(cv=cv, random_state=random_state)

    # Dictionary to store results
    results = {}

    # Fit models and calculate R² scores
    for name, model in [('Lasso', lasso),
                        ('Ridge', ridge),
                        ('ElasticNet', elastic)]:

        fold_predictions = []
        fold_R2s = []
        for train_idx, test_idx in cv.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            # X_train = StandardScaler().fit_transform(X_train)
            # X_test = StandardScaler().fit_transform(X_test)
            y_train, y_test = y[train_idx], y[test_idx]


            # Fit model and make prediction
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            # print(f'\t{r2=:.3f}')
            fold_R2s.append(r2)


        # print(f'Fitting: {name}')
        r2 = np.mean(fold_R2s)

        # Store results
        results[name] = {
            'model': model,
            'r2_score': r2,
            'best_alpha': model.alpha_
        }

        if name == 'ElasticNet':
            results[name]['l1_ratio'] = model.l1_ratio_

    # Generate sample data for demonstration

    # Fit models

    # Print results
    for name, result in results.items():
        print(f"{name} Regression Results:")
        r_score = np.sqrt(result['r2_score'])
        print(f"R² Score: {result['r2_score']:.3f} | {r_score=:.3f}")
        # print(f"Best alpha: {result['best_alpha']:.4f}")
        # if name == 'ElasticNet':
        #     print(f"Best L1 ratio: {result['l1_ratio']:.4f}")

    return results

def get_rissman_similarity_many_layers(pair, cat='attn_weights', get_last=True,
                                       layer_names=None, flip=False,
                                       # activation_model='meta-llama/Llama-2-7b-hf'
                                       activation_model='meta-llama/Llama-3.2-3b'
                                       # activation_model = 'meta-llama/Llama-3.3-70b-Instruct',
                                       ):
    vecs_0_all = []
    vecs_1_all = []
    for layer_name in layer_names:
        kw = {'pair': pair, 'flip': flip, 'cat': cat, 'get_last': get_last,
              'layer_name': layer_name, 'activation_model': activation_model}
        d_vecs_pair = pickle_wrap(get_rissman_similarity,  kwargs=kw,
                                  verbose=-1, easy_override=False)
        if flip:
            vec_0 = d_vecs_pair[(pair[1], pair[0], pair[0])]
            vec_1 = d_vecs_pair[(pair[1], pair[0], pair[1])]
        else:
            vec_0 = d_vecs_pair[(pair[0], pair[1], pair[0])]
            vec_1 = d_vecs_pair[(pair[0], pair[1], pair[1])]
        if cat == 'gate_proj_in':
            prod0 = stats.spearmanr(vec_0, vec_1, nan_policy='omit')[0]
            prod1 = stats.spearmanr(vec_1, vec_0, nan_policy='omit')[0]
            vecs_0_all.append(prod0)
            vecs_1_all.append(prod1)
        else:
            vecs_0_all.extend(vec_0)
            vecs_1_all.extend(vec_1)
    return vecs_0_all, vecs_1_all


def analyze_rissman(cat='attn_weights',
                    # cat='gate_proj_in',
                    get_last=False, layer_name=1):
    pairs, relatedness, df_grp = get_rissman_df()

    M_attns = []
    vecs_attns = []
    for pair in pairs:#, desc='Llama all rissman pairs'):
        if isinstance(layer_name, list):
            vec_010, vec_011 = (
                get_rissman_similarity_many_layers(pair, cat=cat, get_last=get_last,
                                                   layer_names=layer_name, flip=False))
            vec_100, vec_101 = (
                get_rissman_similarity_many_layers(pair, cat=cat, get_last=get_last,
                                                   layer_names=layer_name, flip=True))
        else:
            kw = {'pair': pair, 'flip': False, 'cat': cat, 'get_last': get_last,
                  'layer_name': layer_name}
            d_vecs_pair = pickle_wrap(get_rissman_similarity,  kwargs=kw,
                                      verbose=-1, easy_override=True)
            vec_010 = d_vecs_pair[(pair[0], pair[1], pair[0])]
            vec_011 = d_vecs_pair[(pair[0], pair[1], pair[1])]

            kw['flip'] = True
            d_vecs_pair = pickle_wrap(get_rissman_similarity, kwargs=kw,
                                      verbose=-1)
            vec_100 = d_vecs_pair[(pair[1], pair[0], pair[0])]
            vec_101 = d_vecs_pair[(pair[1], pair[0], pair[1])]

        vecs_pair = np.array([vec_010, vec_011, vec_100, vec_101])
        vec_pair = np.nanmean(vecs_pair, axis=0)
        M_attn = np.nanmean(vec_pair)
        M_attns.append(M_attn)
        vecs_attns.append(vec_pair)

        # TODO: lasso up the multiple weights

    r, p = stats.spearmanr(M_attns, relatedness)
    title = f'{layer_name} | relatedness x mean {cat}: {r=:.2f}, {p=:.3f}'
    print(title)
    # return
    fit_regularized_models(np.array(vecs_attns), relatedness)
    print('-')


if __name__ == '__main__':
    # test_w2v_regression()
    # quit()

    # get_w2v_similarity()
    # quit()
    # get_rissman_w2v_d_vecs()
    analyze_rissman(layer_name=list(range(4, 20)))
    quit()

    for LAYER_NAME in range(0, 24):
        # for FOCUS in range(24):
         analyze_rissman(layer_name=LAYER_NAME)#, focus=FOCUS)