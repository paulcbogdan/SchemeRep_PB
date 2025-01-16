import pickle
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from Utils.pickle_wrap_funcs import pickle_wrap
from llama.devereux_llama import get_llama_activations_deve
from llama.get_obj_scn_vecs import process_cat_cat_inner, get_obj2grammar, get_scn2grammar, get_llama_activations
from llama.plot_2rel import general_llama_plot
from marinate.pkld import pkld
from org_sns import get_sns
from organize_bhv import get_trial_info
from functools import cache
from time import time
import numpy as np
from collections import Counter
from sklearn.utils.validation import check_array, check_X_y, check_random_state

from sklearn.linear_model import (RidgeCV)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     RepeatedKFold, LeaveOneGroupOut,
                                     StratifiedGroupKFold)
from functools import partial
from sklearn.metrics import r2_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.svm import SVC
import warnings

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from collections import Counter
from sklearn.utils.validation import check_array, check_X_y, check_random_state


class BalancedStratifiedKFold(BaseCrossValidator):
    """
    A balanced version of RepeatedStratifiedKFold that ensures perfect stratification
    even with odd numbers of samples.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        # Calculate the ideal distribution per fold
        class_counts = Counter(y)
        total_samples = len(y)

        # Calculate target count per class per fold
        target_counts = {
            label: count // self.n_splits
            for label, count in class_counts.items()
        }

        rng = check_random_state(self.random_state)

        # Repeat the CV n_repeats times
        for _ in range(self.n_repeats):
            # Get initial splits from RepeatedStratifiedKFold
            rskf = RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=1,
                random_state=rng.randint(1000000)
            )

            for _, test_idx in rskf.split(X, y):
                test_idx = np.array(test_idx)
                all_idx = np.arange(len(y))
                train_idx = np.array([i for i in all_idx if i not in test_idx])

                # Check fold distribution
                test_dist = Counter(y[test_idx])

                # If distribution isn't perfect, adjust the split
                for label in class_counts:
                    current_count = test_dist[label]
                    target_count = target_counts[label]

                    while current_count != target_count:
                        if current_count < target_count:
                            # Move one sample from train to test
                            candidates = [i for i in train_idx if y[i] == label]
                            if candidates:
                                idx_to_move = rng.choice(candidates)
                                train_idx = train_idx[train_idx != idx_to_move]
                                test_idx = np.append(test_idx, idx_to_move)
                                current_count += 1
                        else:
                            # Move one sample from test to train
                            candidates = [i for i in test_idx if y[i] == label]
                            if candidates:
                                idx_to_move = rng.choice(candidates)
                                test_idx = test_idx[test_idx != idx_to_move]
                                train_idx = np.append(train_idx, idx_to_move)
                                current_count -= 1

                yield test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Not used in this implementation.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y = check_X_y(X, y, ensure_min_samples=self.n_splits)

        for test_idx in self._iter_test_indices(X, y):
            train_idx = np.array([i for i in range(len(y)) if i not in test_idx])
            yield train_idx, test_idx

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


@pkld
def get_rissman_similarity(pair, flip=False,
                           activation_model='meta-llama/Llama-3.2-3b',
                           cat='attn_weights', get_last=False,
                           layer_name=1, do_schemerep=False):
    cat_, inner = process_cat_cat_inner(cat)
    if do_schemerep:
        item0 = pair[0]
        item1 = pair[1]
    elif flip:
        item0 = pair[1].lower()
        item1 = pair[0].lower()
    else:
        item0 = pair[0].lower()
        item1 = pair[1].lower()

    obj2grammar, _ = get_obj2grammar()
    scn2grammar, _ = get_scn2grammar()
    # print(f'{activation_model=}')
    if do_schemerep:  # pair[0] in obj2grammar and pair[1] in scn2grammar:
        assert not flip
        # print(f'Doing llama: {cat}, {pair=}')
        res, fp = pickle_wrap(get_llama_activations,
                              kwargs={'obj': pair[0], 'scn': pair[1],
                                      'activation_model': activation_model},
                              easy_override=False, verbose=-1, dir_branches=100,
                              RAM_cache=True, get_fp=True
                              )
    else:
        res, fp = pickle_wrap(get_llama_activations_deve,
                              kwargs={'item0': item0, 'item1': item1,
                                      'activation_model': activation_model},
                              easy_override=False, verbose=-1, dir_branches=100,
                              RAM_cache=True, get_fp=True)

    if cat == 'down_proj_out' and not (
            'mlp_out' in res and 'down_proj' in res['mlp_out']):
        if do_schemerep:  # pair[0] in obj2grammar and pair[1] in scn2grammar:
            assert not flip
            print(f'Doing llama (for down_proj): {cat}, {pair=}')
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
                                  easy_override=True, verbose=-1, dir_branches=100,
                                  RAM_cache=False, get_fp=True)

    if 'mlp_out' in res:  # and 'up_proj' in res['mlp_out']:
        if 'up_proj' in res['mlp_out']:
            del res['mlp_out']['up_proj']
        if 'gate_proj' in res['mlp_out']:
            del res['mlp_out']['gate_proj']
        if 'down_proj' in res['mlp_in']:
            del res['mlp_in']['down_proj']
        if 'up_proj' in res['mlp_in']:
            del res['mlp_in']['up_proj']
        if 'act_fn' in res['mlp_in']:
            del res['mlp_in']['act_fn']
        if 'q_proj' in res['attn']:
            del res['attn']['q_proj']
        if 'k_proj' in res['attn']:
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


def plot_best_fit(x, y):
    from statsmodels.formula import api as smf
    df_grp = pd.DataFrame({'x': x, 'y': y})
    mod = smf.ols(formula=f'y ~ 1 + x', data=df_grp)
    res = mod.fit()
    df_pred = pd.DataFrame({'x': [df_grp['x'].min(), df_grp['x'].max()]})
    df_pred['y'] = res.predict(df_pred)
    plt.plot(df_pred['x'], df_pred['y'], color='k', linestyle='--')


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
    plt.figure(figsize=(6, 4))
    plt.rcParams.update({'font.size': 20})
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.scatter(w2v_similarities, relatedness, color='r')
    plot_best_fit(w2v_similarities, relatedness)
    plt.ylim(0.9, 4.1)
    r_sq = r ** 2
    r_sq = .77 * .77
    plt.text(.4, 2.1, f'$r^2$ = {r_sq:.2f}', fontsize=24,
             ha='center')
    # plt.title(f'Relatedness x word2vec similarity: {r=:.2f}')
    plt.xlabel('Word2vec similarity', labelpad=8)
    plt.ylabel('Human-reported\nrelatedness', labelpad=8)
    plt.gcf().subplots_adjust(left=0.175, right=0.975,
                              top=0.975, bottom=0.28)
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


@pkld(overwrite=False)
def fit_regularized_models(X, y, cv_folds=5, random_state=42,
                           plot=False, normalize=False,
                           n_repeats=10, groups=None,
                           do_r2=False, multi_alpha=False):
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


    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Create cross-validation object
    if groups is not None:
        binary = np.unique(y).size == 2
        if len(np.unique(groups)) > 10:
            if binary:
                cv = StratifiedGroupKFold(n_splits=cv_folds)
            else:
                cv = GroupKFold(n_splits=cv_folds)
        else:
            cv = LeaveOneGroupOut()
        cv.split = partial(cv.split, groups=groups)
    elif len(np.unique(y)) < 3:
        cv = BalancedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        # cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=n_repeats,
        #                              random_state=random_state)
        binary = True
    else:
        cv = RepeatedKFold(n_splits=cv_folds, n_repeats=n_repeats,
                           random_state=random_state)
        binary = False
    if isinstance(do_r2, str):
        if do_r2 == 'discrete':
            binary = True
            do_r2 = False
        elif do_r2 == '5050':
            binary = True
            num_ones = (y == 1).sum()
            idxs1 = np.where(y == 1)[0]
            idxs0 = np.where(y == 0)[0]
            idxs0 = idxs0[:num_ones]
            y0 = y[idxs0]
            y1 = y[idxs1]
            x0 = X[idxs0]
            x1 = X[idxs1]
            y = np.concatenate([y0, y1])
            X = np.concatenate([x0, x1])
        else:
            raise ValueError

    if binary:
        model = SVC(kernel='linear', C=1)
    else:
        if multi_alpha:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        else:
            model = RidgeCV()

    fold_accs = []
    coefs = []
    y_tests = []
    y_preds = []
    test_sizes = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        test_sizes.append(len(test_idx))
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if normalize:
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

        # Fit model and make prediction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # print(f'{np.mean(y_train)=:.3f}, {y_train.shape=}')
        # print(f'\t{np.mean(y_test)=:.3f}, {y_test.shape=}')
        if binary:
            if do_r2 == '5050':
                acc = np.mean(y_test == y_pred)
                # print(f'{acc=}')
                # print(y_pred)
                # print(y_test)
                # quit()

                assert .49 < np.mean(y_test) < .51, f'{np.mean(y_test)=}'
            elif do_r2:
                acc = r2_score(y_test, y_pred)

            else:
                acc = np.mean(y_test == y_pred)
        else:
            y_pred[y_pred > 4] = 4
            y_pred[y_pred < 1] = 1
            acc = r2_score(y_test, y_pred)

        fold_accs.append(acc)
        try:
            coefs.append(model.coef_)
        except:
            coefs.append(model.coef0)
        y_tests.extend(list(y_test))
        y_preds.extend(list(y_pred))

    acc = np.average(fold_accs, weights=test_sizes)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.rcParams.update({'font.size': 20})
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.scatter(y_preds, y_tests, color='green')
        plt.title(f'{acc=:.2f}')
        plt.text(3.36, 2.1, f'$r^2$ = {acc:.2f}', fontsize=21,
                 ha='center')
        plt.ylabel('Human-reported\nrelatedness',
                   labelpad=8)
        plt.xlabel('Attention-weight\npredicted relatedness',
                   labelpad=8)
        plt.xlim(0.9, 4.1)
        plt.ylim(0.9, 4.1)
        plt.gcf().subplots_adjust(left=0.175, right=0.975, top=0.975, bottom=0.28)
        plt.show()

    if plot:
        rs_all = []
        for i in range(X.shape[1]):
            r, p = stats.spearmanr(X[:, i], y)
            rs_all.append(r)
        plt.figure(figsize=(6, 4))
        plt.rcParams.update({'font.size': 20})
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.hist(rs_all, bins=20, range=(-1, 1),
                 color='green')
        plt.xticks(np.linspace(-1, 1, 5, ))
        plt.ylabel('Frequency (n)', labelpad=8)
        plt.xlabel('Correlation (r)', labelpad=8)
        plt.gcf().subplots_adjust(left=0.175, right=0.975,
                                  top=0.975, bottom=0.28)
        plt.show()
    return acc


def get_rissman_similarity_many_layers(pair, cat='attn_weights', get_last=True,
                                       layer_names=None, flip=False,
                                       # activation_model='meta-llama/Llama-2-7b-hf'
                                       activation_model='meta-llama/Llama-3.2-3b',
                                       # activation_model = 'meta-llama/Llama-3.3-70b-Instruct',
                                       do_schemerep=False,
                                       ):
    vecs_0_all = []
    vecs_1_all = []
    for layer_name in layer_names:
        kw = {'pair': pair, 'flip': flip, 'cat': cat, 'get_last': get_last,
              'layer_name': layer_name, 'activation_model': activation_model,
              'do_schemerep': do_schemerep}

        d_vecs_pair = get_rissman_similarity(**kw)
        # d_vecs_pair = pickle_wrap(get_rissman_similarity,  kwargs=kw,
        #                           verbose=-1, easy_override=True)
        if do_schemerep:
            pair = (pair[0], pair[1])
        else:
            pair = (pair[0].lower(), pair[1].lower())
        if flip:
            vec_0 = d_vecs_pair[(pair[1], pair[0], pair[0])]
            vec_1 = d_vecs_pair[(pair[1], pair[0], pair[1])]
        else:
            vec_0 = d_vecs_pair[(pair[0], pair[1], pair[0])]
            vec_1 = d_vecs_pair[(pair[0], pair[1], pair[1])]
        if cat != 'attn_weights':
            vecs_0_all.extend(vec_0)
            vecs_1_all.extend(vec_1)
            # prod0 = stats.spearmanr(vec_0, vec_1, nan_policy='omit')[0]
            # prod1 = stats.spearmanr(vec_1, vec_0, nan_policy='omit')[0]
            # vecs_0_all.append(prod0)
            # vecs_1_all.append(prod1)
        else:
            vecs_0_all.extend(vec_0)
            vecs_1_all.extend(vec_1)
    return vecs_0_all, vecs_1_all

def get_SchemeRep_df_cont(no_neu):
    pairs, _, _ = get_SchemeRep_df(no_neu)
    d_l = {}
    for pair in pairs:
        d_l[pair] = []
    sns = get_sns('all')['healthy']
    for sn in sns:
        df = get_trial_info(sn)
        for i, row in df.iterrows():
            pair = (row['obj'], row['scene'])
            if pair in d_l:
                d_l[pair].append(row['per_inc'])
    relatedness = [np.nanmean(d_l[pair]) for pair in pairs]
    return pairs, relatedness, None

@cache
def get_SchemeRep_df(no_neu=False):
    sns = ['102', '103', '104']  # everyone else is a duplicate
    already_done = set()
    objs_l = []
    scns_l = []
    ics_l = []
    for sn in sns:
        df_sn = get_trial_info(sn)
        if isinstance(no_neu, int) or isinstance(no_neu, str):
            no_neu = int(no_neu)
            df_sn = df_sn[df_sn['inc'] != no_neu]
        # elif isinstance(no_neu, tuple):
        #     df_sn = df_sn[df_sn['inc'] != no_neu]
        elif no_neu:
            df_sn = df_sn[df_sn['inc'] != 2]
        objs = df_sn['obj'].to_list()
        scns = df_sn['scene'].to_list()
        ics = df_sn['inc'].to_list()
        objs_l.extend(objs)
        scns_l.extend(scns)
        ics_l.extend(ics)
    pairs = list(zip(objs_l, scns_l))
    print(f'Prepped SchemeRep pairs: {pairs=}')
    return pairs, ics_l, None


def normalize_rissman_SchemeRep(pairs, vecs):
    obj2l = defaultdict(list)
    for pair, vec in zip(pairs, vecs):
        obj2l[pair[0]].append(vec)
    obj2M = {}
    for obj, vecs_pair in obj2l.items():
        M = np.nanmean(vecs_pair, axis=0)
        obj2M[obj] = M
    for i, pair in enumerate(pairs):
        vecs[i] -= obj2M[pair[0]]
    return vecs


@pkld(overwrite=False)
def analyze_rissman(cat='attn_weights',
                    # cat='gate_proj_in',
                    activation_model='meta-llama/Llama-3.2-3b',
                    get_last=False, layer_name: Union[int, list] = 1,
                    do_SchemeRep=False, plot_hist=False,
                    norm_SchemeRep=False,
                    binary_nonrep=False,
                    no_neu=False,
                    # get_vecs=False
                    ):


    if isinstance(activation_model, list):
        if activation_model[1] == 'get_vecs':
            get_vecs = True
            activation_model = activation_model[0]
        else:
            _, relatedness, _ = get_rissman_df()
            vecs_all = []
            for activation_model_ in activation_model:
                vecs = analyze_rissman(cat=cat, activation_model=[activation_model_, 'get_vecs'],
                                          get_last=get_last, layer_name=layer_name,
                                          do_SchemeRep=do_SchemeRep, plot_hist=plot_hist,
                                          norm_SchemeRep=norm_SchemeRep,
                                          binary_nonrep=binary_nonrep, no_neu=no_neu)
                vecs_all.append(vecs)
                vecs = np.array(vecs)
                vecs = stats.zscore(vecs, axis=0, nan_policy='omit')
                result = fit_regularized_models(vecs, relatedness, plot=plot_hist,
                                                n_repeats=1, normalize=False,
                                                groups=groups if do_SchemeRep else None)
                print(f'{activation_model_} | {result["Ridge"]["r2_score"]=:.2f}')

            vecs_all = np.concatenate(vecs_all, axis=1)

            result = fit_regularized_models(vecs_all, relatedness, plot=plot_hist,
                                            n_repeats=1, normalize=False,
                                            groups=groups if do_SchemeRep else None)
            print(f'{activation_model} | {result["Ridge"]["r2_score"]=:.2f}')

            result = fit_regularized_models(np.array(vecs_all), relatedness, plot=plot_hist,
                                            n_repeats=1, normalize=False,
                                            groups=groups if do_SchemeRep else None)
            title = f'{layer_name} | {result["Ridge"]["r2_score"]=:.2f}'
            print(title)

            r2 = result['Ridge']['r2_score']
            return r2
    else:
        get_vecs = False


    assert not (binary_nonrep and do_SchemeRep)
    if do_SchemeRep:
        if isinstance(no_neu, tuple) and no_neu[1] == 'cont':
            pairs, relatedness, _ = get_SchemeRep_df_cont(no_neu[0])

        else:
            pairs, relatedness, _ = get_SchemeRep_df(no_neu=no_neu)
        if no_neu and not (isinstance(no_neu, tuple) and no_neu[0] == False):

            t_st = time()
            print('Finding circles...')
            circles = find_circles(pairs)
            print(f'{time() - t_st=:.2f} s')

            word2circle = {}
            for i, circle in enumerate(circles):
                for word in circle:
                    word2circle[word] = i
                    word2circle[word[1]] = i
            groups = [word2circle[pair[0]] for pair in pairs]
            for pair, g in zip(pairs, groups):
                assert word2circle[pair[0]] == g
                assert word2circle[pair[1]] == g
        else:
            groups = [pair[0] for pair in pairs]
    else:
        pairs, relatedness, _ = get_rissman_df()
    M_attns = []
    vecs_attns = []
    do_SchemeRep_groups = do_SchemeRep
    if isinstance(do_SchemeRep, tuple) and do_SchemeRep[1] == 'deve':
        do_SchemeRep = False

    # quit()
    for pair in pairs:
        # print('Getting pairs')
        if isinstance(layer_name, list):
            vec_010, vec_011 = (
                get_rissman_similarity_many_layers(pair, cat=cat, get_last=get_last,
                                                   layer_names=layer_name, flip=False,
                                                   do_schemerep=do_SchemeRep,
                                                   activation_model=activation_model))
            if not do_SchemeRep:
                vec_100, vec_101 = (
                    get_rissman_similarity_many_layers(pair, cat=cat, get_last=get_last,
                                                       layer_names=layer_name, flip=True,
                                                       do_schemerep=do_SchemeRep,
                                                       activation_model=activation_model))
        else:
            kw = {'pair': pair, 'flip': False, 'cat': cat, 'get_last': get_last,
                  'layer_name': layer_name, 'do_schemerep': do_SchemeRep,
                  'activation_model': activation_model}
            # d_vecs_pair = pickle_wrap(get_rissman_similarity,  kwargs=kw,
            #                           verbose=-1, easy_override=False)
            # d_vecs_pair = get_rissman_similarity(**kw)
            # print(f'Getting pair: {pair}')
            d_vecs_pair = pkld(get_rissman_similarity, overwrite=False,
                               verbose=0)(**kw)

            # print('test')
            # quit()
            # d_vecs_pair = pkld(get_rissman_similarity(**kw))()

            if do_SchemeRep:
                pair = (pair[0], pair[1])
            else:
                pair = (pair[0].lower(), pair[1].lower())
            vec_010 = d_vecs_pair[(pair[0], pair[1], pair[0])]
            vec_011 = d_vecs_pair[(pair[0], pair[1], pair[1])]

            if not do_SchemeRep:
                kw['flip'] = True
                d_vecs_pair = get_rissman_similarity(**kw)
                # d_vecs_pair = pickle_wrap(get_rissman_similarity, kwargs=kw,
                #                           verbose=-1, easy_override=False)
                vec_100 = d_vecs_pair[(pair[1], pair[0], pair[0])]
                vec_101 = d_vecs_pair[(pair[1], pair[0], pair[1])]


        if do_SchemeRep:
            if cat != 'attn_weights':
                vecs_pair = np.array([vec_010])  # this is the correct order, 0 = obj, 1 = scn
            else:
                vecs_pair = np.array([vec_010, vec_011])
        else:
            vecs_pair = np.array([vec_010, vec_011, vec_100, vec_101])

        vec_pair = np.nanmean(vecs_pair, axis=0)
        M_attn = np.nanmean(vec_pair)
        M_attns.append(M_attn)
        vecs_attns.append(vec_pair)

        # TODO: lasso up the multiple weights

    r, p = stats.spearmanr(M_attns, relatedness)
    relatedness = np.array(relatedness)
    vecs_attns = np.array(vecs_attns)

    if norm_SchemeRep:
        print('Normalizing rissman SchemeRep')
        vecs_attns = normalize_rissman_SchemeRep(pairs, vecs_attns)

    if binary_nonrep:
        relatedness = np.array(relatedness)
        relatedness[relatedness < 3] = 0
        relatedness[relatedness >= 3] = 1

    if get_vecs:
        return np.array(vecs_attns)

    # from time import time
    # t_st = time()
    result = fit_regularized_models(np.array(vecs_attns), relatedness, plot=plot_hist,
                                    n_repeats=1, normalize=False,
                                    groups=groups if do_SchemeRep_groups else None)
    title = f'{layer_name} | mean {cat}: {r=:.2f}, {result["Ridge"]["r2_score"]=:.2f}'
    print(title)
    # print(f'{time() - t_st=:.2f} s')
    # quit()

    r2 = result['Ridge']['r2_score']
    return r2


def compare_attn_vs_gate(do_SchemeRep=False, norm_SchemeRep=False,
                         # activation_model='meta-llama/Llama-3.2-3b',
                         activation_model=('meta-llama/Llama-3.2-3b', 'bury'),
                         binary_nonrep=True, no_neu=False,
                         do_xlabel=True, four_piece=False
                         # activation_model='meta-llama/Llama-3.3-70b-Instruct',
                         ):
    print(f'Plotting compare_attn_vs_gate: {do_SchemeRep}, {no_neu}: {activation_model=}')
    # .25, .6,
    #     all_llama_cats = ['gate_proj_in', 'up_proj_in', 'down_proj_in', 'act_fn_in',
    #                       'gate_proj_out', 'up_proj_out', 'down_proj_out', 'act_fn_out',
    #                       'q_proj', 'k_proj', 'v_proj', 'attn_weights', '| tput',
    #                       'input']

    assert not ((not do_SchemeRep) and (norm_SchemeRep))
    assert not (no_neu and not do_SchemeRep)
    vals_attn = []
    vals_residual = []
    vals_attn_output = []
    vals_down = []
    vals_mid = []
    num_layers = 28 if ('3b' in activation_model or '3b' in activation_model[0]) else 80
    for layer_name in range(num_layers):
        # layer_name_l = list(range(layer_name, layer_name + 4))
        layer_name_l = layer_name
        kw = {'layer_name': layer_name_l, 'do_SchemeRep': do_SchemeRep,
              'activation_model': activation_model, 'norm_SchemeRep': norm_SchemeRep,
              'binary_nonrep': binary_nonrep, 'no_neu': no_neu}

        if no_neu == 'all':
            r2_attn, r2_input, r2_mid, r2_attn_output, r2_down = (
                [], [], [], [], [])
            for nn in range(1, 4):
                if nn == 2: continue
                kw['no_neu'] = nn
                r2_attn.append(analyze_rissman(cat='attn_weights', **kw))
                r2_input.append(analyze_rissman(cat='input', **kw))
                r2_mid.append(analyze_rissman(cat='gate_proj_in', **kw))
                r2_attn_output.append(analyze_rissman(cat='attn_output', **kw))
                r2_down.append(analyze_rissman(cat='down_proj_out', **kw))
            r2_attn = np.mean(r2_attn, axis=0)
            r2_input = np.mean(r2_input, axis=0)
            r2_mid = np.mean(r2_mid, axis=0)
            r2_attn_output = np.mean(r2_attn_output, axis=0)
            r2_down = np.mean(r2_down, axis=0)
        else:
            r2_attn = analyze_rissman(cat='attn_weights', **kw)
            r2_input = analyze_rissman(cat='input', **kw)
            r2_mid = analyze_rissman(cat='gate_proj_in', **kw)
            r2_attn_output = analyze_rissman(cat='attn_output', **kw)
            r2_down = analyze_rissman(cat='down_proj_out', **kw)

        print(f'{layer_name} | {r2_attn=:.2f}, {r2_input=:.2f}, {r2_mid=:.2f}, '
              f'{r2_attn_output=:.2f}, {r2_down=:.2f}')
        vals_attn.append(r2_attn)
        vals_residual.append(r2_input)
        vals_attn_output.append(r2_attn_output)
        vals_down.append(r2_down)
        vals_mid.append(r2_mid)

    # plt.plot(list(range(len(vals_attn))), vals_attn,
    #          label='Attention', color='green', marker='.')
    # plt.plot(list(range(len(vals_residual))), vals_residual,
    #          label='Residual (input)', color='purple', marker='.',
    #          alpha=0.5)
    # plt.plot(list(range(len(vals_mid))), vals_mid,
    #          label='Residual (middle)', color='k', marker='.',
    #          alpha=0.5)
    # plt.plot(list(range(len(vals_attn_output))), vals_attn_output,
    #          label='Attention addition', color='red', marker='.',
    #          alpha=0.5)
    # plt.plot(list(range(len(vals_down))), vals_down,
    #          label='MLP addition', color='dodgerblue', marker='.',
    #          alpha=0.5)
    #
    # r_attn = get_autocorr(vals_attn)
    # r_input = get_autocorr(vals_residual)
    # r_attn_out = get_autocorr(vals_attn_output)
    # r_FFN = get_autocorr(vals_down)
    # r_mid = get_autocorr(vals_mid)
    # autocorr_title = (f'{r_attn=:.2f}, {r_input=:.2f}, {r_attn_out=:.2f}, '
    #                   f'{r_FFN=:.2f}, {r_mid=:.2f}')
    # plt.xlabel('Layer')
    # plt.legend()
    plt.rcParams.update({'font.size': 14})
    if isinstance(no_neu, tuple) and no_neu[1] == 'cont':
        if isinstance(do_SchemeRep, tuple) and 'deve' in do_SchemeRep:
            if four_piece:
                title = 'Object-scene\ncongruence'
            else:
                title = 'Object-scene congruence\n"An {object} and {scene}"'
        else:
            if four_piece:
                title = 'Object-scene\ncongruence'
            else:
                title = 'Object-scene congruence\n"In the {scene}, an {object}"'
        y_low = 0
        do_ylabel = 'Accuracy (R²)'
        # plt.ylim(0, 1)
    elif (do_SchemeRep and no_neu ) or binary_nonrep:
        title = 'Object-scene congruence binary'
        y_low = 0.5
        do_ylabel = 'Accuracy (%)'
        # plt.ylim(0.5, 1)
    else:
        if four_piece:
            title = 'Object-object\nrelatedness'
        else:
            title = 'Object-object relatedness\n"An {item 1} and {item 2}"'
        # plt.title('Object-object relatedness\n"An {item 1} and {item 2}"',
        #           fontsize=16)
        do_ylabel = 'Accuracy (R²)'
        y_low = 0
    plt.title(title, fontsize=16)
    vals_l = [vals_residual, vals_attn_output, vals_down,
              #vals_attn
              ]
    labels = ['Residual\n(input)', 'Attention\naddition',
              'FFN\naddition', #'Attention\nweights'
              ]
    colors = ['k', 'r', 'dodgerblue',
              # 'green'
              ]
    # from time import time
    # t_st = time()
    general_llama_plot(vals_l, labels, colors, ylabel=do_ylabel,
                       y_low=y_low, do_legend=False, xlabel=do_xlabel,
                       num_layers=num_layers, do_ylabel=not four_piece)


def get_autocorr(l, gap=1):
    l = l[5:]
    dif0 = np.diff(l)
    return stats.spearmanr(dif0[:-gap], dif0[gap:], nan_policy='omit')[0]

def find_circles(pairs):
    # Create adjacency dict
    adj = {}
    for i, j in pairs:
        if i not in adj:
            adj[i] = []
        if j not in adj:
            adj[j] = []
        adj[i].append(j)
        adj[j].append(i)


    def dfs(node, parent, path, circles):
        if node in path:
            # Found circle - get the circle portion
            circle = path[path.index(node):]
            circles.append(circle)
            return

        path.append(node)
        for neighbor in adj[node]:
            if neighbor != parent:
                dfs(neighbor, node, path, circles)
        path.pop()

    circles = []
    visited = set()

    # Start DFS from each node
    for node in adj:
        if node not in visited:
            dfs(node, None, [], circles)
            visited.add(node)

    for circle in circles:
        circle.sort()
    circles = list(set(tuple(circle) for circle in circles))

    return circles


def try_concat(do_SchemeRep=False, norm_SchemeRep=False,
             # activation_model='meta-llama/Llama-3.2-3b',
             activation_model=('meta-llama/Llama-3.2-3b', 'bury'),
             binary_nonrep=False, no_neu=False,):
    r2 = analyze_rissman(activation_model=['meta-llama/Llama-3.2-3b',
                                      ('meta-llama/Llama-3.2-3b', 'bury')],
                    layer_name=[7, 17],
                    cat='input',
                    do_SchemeRep=do_SchemeRep,
                    )
    print(f'{r2=:.5f}')

    r2 = analyze_rissman(activation_model='meta-llama/Llama-3.2-3b',
                    layer_name=[7, 17],
                    cat='input',
                    do_SchemeRep=do_SchemeRep,
                    )
    print(f'{r2=:.5f}')


    r2 = analyze_rissman(activation_model=('meta-llama/Llama-3.2-3b', 'bury'),
                    layer_name=[7, 17],
                    cat='input',
                    do_SchemeRep=do_SchemeRep,
                    )
    print(f'{r2=:.5f}')

    quit()

def prep_all_figures():
    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
                         do_SchemeRep=(True, 'deve'), norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=1)
    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
                         do_SchemeRep=True, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=1)
    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=False, no_neu=False)

    compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.2-3b', 'bury'),
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=True, no_neu=False)
    quit()

    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
    #                      do_SchemeRep=(True, 'deve'), norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=1)
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
    #                      do_SchemeRep=(True, 'deve'), norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=3)
    # quit()

    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=True, no_neu=False)


    compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.2-3b', ('bury', 0.25)),
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=True, no_neu=False)

def prep_all_figures70():
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=True, norm_SchemeRep=False, # TODO: toggle back to True
    #                      binary_nonrep=False, no_neu=1)
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=True, norm_SchemeRep=False,
    #                      binary_nonrep=False, no_neu=3)
    # # quit()
    #
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=True, norm_SchemeRep=False,
    #                      binary_nonrep=False, no_neu=2)
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=(True, 'deve'), norm_SchemeRep=False,
    #                      binary_nonrep=False, no_neu=3)
    #
    #
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=False, norm_SchemeRep=False,
    #                      binary_nonrep=True, no_neu=False)
    # quit()
    # for no_neu in range(1, 4):
    compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu='all')
    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu='all')
    compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
                         do_SchemeRep=False, norm_SchemeRep=False,
                         binary_nonrep=True, no_neu=False)

    # TODO: double-check that no_rep 1 is actually dropping inc

def investigate_70_bury(#do_SchemeRep=(True, 'deve'),
                        activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
                        do_SchemeRep=True
                        ):
    compare_attn_vs_gate(activation_model=activation_model,
                         do_SchemeRep=do_SchemeRep, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu=(False, 'cont'))
    # compare_attn_vs_gate(activation_model=activation_model,
    #                      do_SchemeRep=do_SchemeRep, norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=(1, 'cont'))
    # compare_attn_vs_gate(activation_model=activation_model,
    #                      do_SchemeRep=do_SchemeRep, norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=(2, 'cont'))
    # compare_attn_vs_gate(activation_model=activation_model,
    #                      do_SchemeRep=do_SchemeRep, norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=(3, 'cont'))
    # return
    #
    # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
    #                      do_SchemeRep=False, norm_SchemeRep=False,
    #                      binary_nonrep=False, no_neu=False)
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
    #                      do_SchemeRep=False, norm_SchemeRep=False,
    #                      binary_nonrep=False, no_neu=False)

    compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu='all')
    compare_attn_vs_gate(activation_model='meta-llama/Llama-3.3-70b-Instruct',
                         do_SchemeRep=True, norm_SchemeRep=True,
                         binary_nonrep=False, no_neu='all')

def print_all_sentences():
    pass



if __name__ == '__main__':
    # TODO: test the SchemeRep ws ile doing the "and" format
    #   do_SchemeRep=(True, 'deve')
    # prep_all_figures()
    # TODO: compare analogy 70
    # prep_all_figures70()
    investigate_70_bury(activation_model='meta-llama/Llama-3.3-70b-Instruct')
    investigate_70_bury(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'))


    # investigate_70_bury(activation_model='meta-llama/Llama-3.2-3b')
    # investigate_70_bury(activation_model=('meta-llama/Llama-3.2-3b', 'bury'))

    quit()

    # quit()
    # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'),
    #                      do_SchemeRep=False, norm_SchemeRep=False,
    #                      binary_nonrep=True, no_neu=False)

    # prep_all_figures()
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
    #                      do_SchemeRep=False, norm_SchemeRep=False,
    #                      binary_nonrep=True, no_neu=False)
    #
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b',
    #                      do_SchemeRep=True, norm_SchemeRep=True,
    #                      binary_nonrep=False, no_neu=True)
    # quit()
    # try_concat()
    # pair = ('apple', 'bank')
    # res, fp = pickle_wrap(get_llama_activations,
    #                       kwargs={'obj': pair[0], 'scn': pair[1],
    #                               'activation_model': 'meta-llama/Llama-3.2-3b'},
    #                       easy_override=False, verbose=-1, dir_branches=100,
    #                       RAM_cache=False, get_fp=True
    #                       )
    #
    #
    # a = res['attn']['attn_output'][0][0][0]
    #
    # for layer in range(24):
    #
    #     b0 = res['mlp_in']['gate_proj'][layer][0][0]
    #     # b_attn = res['attn']['attn_output'][layer][0][0]
    #     # b0 += res['mlp_out']['down_proj'][layer][0][0]
    #
    #     b1 = res['attn']['input'][layer + 1][0][0]# + b_attn
    #
    #     r, p = stats.spearmanr(b1, b0, nan_policy='omit')
    #     print(f'{layer} | {r=:.3f}, {p=:.3f}')
    # quit()
    # # # # print(res['mlp_out']['gate_proj'][0])

    # print(res['attn']['attn_output'][0])

    # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.2-3b', 'top'))
    # compare_attn_vs_gate(activation_model='meta-llama/Llama-3.2-3b')#('meta-llama/Llama-3.2-3b', 'top'))
    #
    #
    #
    # quit()
    #
    #
    # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', 'bury'))
    # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', ('bury', 0.25)))
    # # compare_attn_vs_gate(activation_model=('meta-llama/Llama-3.3-70b-Instruct', ('bury', 0.60)))
    #
    # quit()

    # for LAYER_NAME in range(0, 24):
    #     # for FOCUS in range(24):
    #     analyze_rissman(layer_name=LAYER_NAME)  # , focus=FOCUS)
