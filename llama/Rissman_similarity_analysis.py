import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
from time import time

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve
from llama.get_obj_scn_vecs import process_cat_cat_inner, get_obj2grammar, get_scn2grammar, get_llama_activations
from organize_bhv import get_trial_info


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

    obj2grammar, _ = get_obj2grammar()
    scn2grammar, _ = get_scn2grammar()
    if pair[0] in obj2grammar and pair[1] in scn2grammar:
        res, fp = pickle_wrap(get_llama_activations,
                                  kwargs={'obj': pair[0], 'scn': pair[1],
                                          'activation_model': activation_model},
                                  easy_override=False, verbose=-1, dir_branches=100,
                                  RAM_cache=False, get_fp=True
                                  )
    else:
        res, fp = pickle_wrap(get_llama_activations_deve,
                              kwargs={'item0': item0, 'item1': item1,
                                      'activation_model': activation_model},
                              easy_override=False, verbose=-1, dir_branches=100,
                              RAM_cache=False, get_fp=True)

    if 'mlp_out' in res and 'down_proj' in res['mlp_out']:
        del res['mlp_out']['down_proj']
        del res['mlp_out']['up_proj']
        del res['mlp_out']['gate_proj']
        del res['mlp_in']['down_proj']
        del res['mlp_in']['up_proj']
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
    # print(list(d_vecs.keys()))
    # quit()
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

def fit_regularized_models(X, y, cv_folds=5, random_state=42,
                           plot=False):

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
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.svm import SVC
    import warnings
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_scaled = X

    # Create cross-validation object
    if len(np.unique(y)) <= 3:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                             random_state=random_state)
        binary = True
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True,
                             random_state=random_state)
        binary = False
    # cv = LeaveOneOut()
    # print(f'{binary=}')

    # Initialize models
    lasso = LassoCV(cv=cv, random_state=random_state)
    ridge = RidgeCV(cv=cv)
    ridge = RidgeCV(cv=cv)
    svm = SVC(kernel='linear', C=1)
    elastic = ElasticNetCV(cv=cv, random_state=random_state)

    # Dictionary to store results
    results = {}

    # Fit models and calculate R² scores
    for name, model in [#('Lasso', lasso),
                        # ('Ridge', ridge),
                        ('Ridge', svm) if binary else ('Ridge', ridge),
                        #('ElasticNet', elastic)
                        ]:
        # print(f'{X=}')
        # print(f'{y=}')

        fold_predictions = []
        fold_R2s = []
        coefs = []
        y_tests = []
        y_preds = []
        for train_idx, test_idx in cv.split(X, y):
            # Split data

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # print(f'{y_train=}')
            # Fit model and make prediction
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred[y_pred > 4] = 4
            y_pred[y_pred < 1] = 1
            r2 = r2_score(y_test, y_pred)
            fold_R2s.append(r2)
            coefs.append(model.coef_)

            y_tests.extend(list(y_test))
            y_preds.extend(list(y_pred))


        coefs = np.mean(coefs, axis=0)
        r2 = np.mean(fold_R2s)

        # Store results
        results[name] = {
            'model': model,
            'r2_score': r2,
            # 'best_alpha': model.alpha_,
            'coefs': coefs
        }

        if name == 'ElasticNet':
            results[name]['l1_ratio'] = model.l1_ratio_

        if plot or True:
            plt.scatter(y_tests, y_preds)
            plt.title(f'{name} | {r2=:.2f}')
            plt.show()

    # Print results
    for name, result in results.items():
        # print(f"{name} Regression Results:")
        if result['r2_score'] < 0:
            r_score = 0
        else:
            r_score = np.sqrt(result['r2_score'])
        # print(f"R² Score: {result['r2_score']:.3f} | {r_score=:.3f}")
        rs_all = []
        for i in range(len(result['coefs'])):
            r, p = stats.spearmanr(X[:, i], y)
            rs_all.append(r)
        plt.hist(rs_all, bins=20, range=(-1, 1),
                 color='dodgerblue')
        plt.title(f'{name} | {r_score=:.2f}')
        plt.xticks(np.linspace(-1, 1, 11,))
        plt.ylabel('Frequency (n)')
        plt.xlabel('Correlation (r)')
        plt.show()
        quit()
            # print(f'\t{i} | {r=:.2f}, {result["coefs"][i]=:.2f}')

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
                                  verbose=-1, easy_override=True)
        pair = (pair[0].lower(), pair[1].lower())
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

def get_SchemeRep_df(no_neu=True):
    sns = ['102', '103', '104'] # everyone else is a duplicate
    already_done = set()
    objs_l = []
    scns_l = []
    ics_l = []
    for sn in sns:
        df_sn = get_trial_info(sn)
        if no_neu:
            df_sn = df_sn[df_sn['inc'] != 2]
        objs = df_sn['obj'].to_list()
        scns = df_sn['scene'].to_list()
        ics = df_sn['inc'].to_list()
        objs_l.extend(objs)
        scns_l.extend(scns)
        ics_l.extend(ics)
    pairs = list(zip(objs_l, scns_l))
    return pairs, ics_l, None

def analyze_rissman(cat='attn_weights',
                    #cat='gate_proj_in',
                    get_last=False, layer_name=1,
                    do_SchemeRep=False):
    if do_SchemeRep:
        pairs, relatedness, _ = get_SchemeRep_df()
    else:
        pairs, relatedness, _ = get_rissman_df()

    M_attns = []
    vecs_attns = []
    for pair in pairs:
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
            pair = (pair[0].lower(), pair[1].lower())
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
    relatedness = np.array(relatedness)
    vecs_attns = np.array(vecs_attns)
    # for i in range(vecs_attns.shape[1]):
    #     r, p = stats.spearmanr(vecs_attns[:, i], relatedness)
    #     plt.scatter(vecs_attns[:, i], relatedness)
    #     plt.title(f'{i} | {r=:.2f}')
    #     plt.show()
    #     print(f'{i}: {r=:.3f}, {p=:.3f}')
    # print(vecs_attns.shape)
    # quit()


    fit_regularized_models(np.array(vecs_attns), relatedness)
    print('-')


if __name__ == '__main__':
    # test_w2v_regression()
    # quit()

    # get_w2v_similarity()
    # quit()
    # get_rissman_w2v_d_vecs()
    # analyze_rissman(layer_name=8)

    # analyze_rissman(layer_name=list(range(4, 20)))
    analyze_rissman(layer_name=list(range(8, 16)))
    quit()

    for LAYER_NAME in range(0, 24):
        # for FOCUS in range(24):
         analyze_rissman(layer_name=LAYER_NAME)#, focus=FOCUS)