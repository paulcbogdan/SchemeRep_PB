import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from llama.get_obj_scn_vecs import get_sn_fp_llama_RSM
from scipy import stats
import pandas as pd

def run_RSM_x_RSM():
    # activation_model = 'meta-llama/Llama-3.2-3b'
    # activation_model = 'meta-llama/Llama-3.1-70b'
    activation_model = 'meta-llama/Llama-3.3-70b-Instruct'

    normalize = True
    all_llama_layers = list(range(0, 80 if '70b' in activation_model else 28))
    all_llama_cats = ['gate_proj_in']#, 'attn_weights']
    # all_llama_cats = ['attn_weights']
    semantic_l = []
    cnt = 0
    tick_lows = []
    tick_mids = []
    RSM_stims_all = []
    RSM_stims_all_ar = []
    ticks = []
    for llama_layer in all_llama_layers:
        tick_lows.append(cnt)
        ticks.append(llama_layer)
        for llama_cat in all_llama_cats:
            semantic = ('llama', llama_cat, llama_layer, 'obj', activation_model,
                        normalize)
            print(f'{semantic=}')
            RSM_stim = get_sn_fp_llama_RSM(104, 'obj7_fMRI', semantic)
            RSM_stims_all_ar.append(RSM_stim)

            RSM_stim = RSM_stim[np.tril_indices_from(RSM_stim, k=-1)]
            RSM_stims_all.append(RSM_stim)
            cnt += 1
        tick_mid = (cnt - tick_lows[-1]) // 2 + tick_lows[-1]
        tick_mids.append(tick_mid)


    RSM_stims_all = np.array(RSM_stims_all)
    nan_cols = np.isnan(RSM_stims_all).all(axis=0)
    RSM_stims_all_no_nan = RSM_stims_all[:, ~nan_cols]

    plt.figure(figsize=(12, 12))
    # set default font size to 16
    plt.rcParams.update({'font.size': 18})
    RSM_stims_all_r = stats.rankdata(RSM_stims_all_no_nan, axis=1)
    corr = spatial.distance.pdist(RSM_stims_all_r, 'correlation')
    corr = spatial.distance.squareform(corr)
    corr[np.diag_indices_from(corr)] = np.nan
    corr = 1 - corr
    plt.imshow(corr, cmap='turbo', vmin=0.6, vmax=1.0, interpolation='none')

    if len(tick_mids) > 20:
        tick_mids = tick_mids[::5]
        ticks = ticks[::5]

    # plt.title()
    plt.xticks(tick_mids, ticks, rotation=90, fontsize=16)
    plt.xlabel('Layer')
    plt.yticks(tick_mids, ticks, rotation=0, fontsize=16)
    plt.ylabel('Layer')
    ax = plt.colorbar()
    ax.set_label('Spearman (r)')
    plt.title('RSM x RSM between layers\' MLP outputs')
    # plt.tight_layout()
    plt.show()
    quit()

    cluster_RSMs(RSM_stims_all_no_nan, RSM_stims_all_no_nan)
    cluster_RSMs2(RSM_stims_all_no_nan)



    return semantic_l
    
def do_PCA(RSM_stims_all):
    RSM_stims_all = np.array(RSM_stims_all)
    from sklearn.decomposition import PCA
    RSM_stims_all = stats.zscore(RSM_stims_all, axis=1, nan_policy='omit')
    pca = PCA()
    pca_result = pca.fit_transform(RSM_stims_all)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot: Variance Explained by Each Principal Component')
    plt.tight_layout()
    plt.show()


def cluster_RSMs(data, scaled_data, n_clusters=8):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score



    data = pd.DataFrame(data.T)


    corr_matrix = np.corrcoef(scaled_data)
    # Compute distance matrix from correlation
    distance_matrix = 1 - np.abs(corr_matrix)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # distance_matrix = corr_matrix

    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='ward', optimal_ordering=True)

    # Cut the dendrogram into clusters
    from scipy.cluster.hierarchy import fcluster
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

    # Calculate cluster contributions
    cluster_contributions = {}

    for cluster in range(1, n_clusters + 1):
        cluster_columns = data.columns[cluster_labels == cluster]

        # Compute cluster centroids
        cluster_centroid = data[cluster_columns].mean(axis=1)

        # Calculate contribution score for each column in the cluster
        cluster_contributions[cluster] = {}
        for col in cluster_columns:
            # Compute correlation of column with cluster centroid
            contribution_score = np.abs(np.corrcoef(data[col], cluster_centroid)[0, 1])
            cluster_contributions[cluster][col] = contribution_score

    # Visualize dendrogram
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    dendrogram(linkage_matrix, labels=data.columns, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering of Columns')
    plt.tight_layout()
    plt.show()

    # Create summary dataframe of contributions
    contribution_df = pd.DataFrame.from_dict({
        (cluster, col): score
        for cluster, cols in cluster_contributions.items()
        for col, score in cols.items()
    }, orient='index', columns=['Contribution Score'])
    contribution_df.index = pd.MultiIndex.from_tuples(contribution_df.index, names=['Cluster', 'Column'])

    # Visualize contribution scores
    plt.figure(figsize=(12, 8))
    # quit()
    plt.rcParams.update({'font.size': 10})

    contribution_df.unstack().T.plot(kind='bar', stacked=True)
    plt.title('Column Contributions Across Clusters')
    plt.xlabel('Columns')
    plt.ylabel('Contribution Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    quit()

def cluster_RSMs2(data, max_clusters=10):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    data = pd.DataFrame(data.T)


    # Ensure we're working with the correlation matrix of columns
    if data.shape[0] != data.shape[1]:
        # If data is not square, compute correlation matrix of columns
        corr_matrix = data.corr().abs()
    else:
        corr_matrix = np.abs(data)
    corr_matrix = np.array(corr_matrix)

    # Ensure matrix is symmetric
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(corr_matrix, 0)

    # Prepare distance for clustering
    condensed_dist = squareform(distance_matrix)

    # Methods for determining optimal clusters
    methods = {
        'Silhouette Score': [],
        'Calinski-Harabasz Score': [],
        'Davies-Bouldin Score': []
    }

    # Test different numbers of clusters
    cluster_range = range(2, min(max_clusters + 1, len(corr_matrix)))

    for n_clusters in cluster_range:
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='ward')
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

        # Compute metrics
        # Use the correlation matrix for these calculations
        try:
            methods['Silhouette Score'].append(
                silhouette_score(corr_matrix, cluster_labels, metric='precomputed')
            )
            methods['Calinski-Harabasz Score'].append(
                calinski_harabasz_score(corr_matrix, cluster_labels)
            )
            methods['Davies-Bouldin Score'].append(
                davies_bouldin_score(corr_matrix, cluster_labels)
            )
        except Exception as e:
            print(f"Error computing metrics for {n_clusters} clusters: {e}")
            break
    # quit()

    # Plotting
    plt.figure(figsize=(15, 5))

    # Silhouette Score (higher is better)
    plt.subplot(131)
    plt.plot(list(cluster_range)[:len(methods['Silhouette Score'])],
             methods['Silhouette Score'])
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    # Calinski-Harabasz Score (higher is better)
    plt.subplot(132)
    plt.plot(list(cluster_range)[:len(methods['Calinski-Harabasz Score'])],
             methods['Calinski-Harabasz Score'])
    plt.title('Calinski-Harabasz Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    # Davies-Bouldin Score (lower is better)
    plt.subplot(133)
    plt.plot(list(cluster_range)[:len(methods['Davies-Bouldin Score'])],
             methods['Davies-Bouldin Score'])
    plt.title('Davies-Bouldin Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

    # Determine optimal number of clusters
    optimal_clusters = {
        'Silhouette': list(cluster_range)[np.argmax(methods['Silhouette Score'])],
        'Calinski-Harabasz': list(cluster_range)[np.argmax(methods['Calinski-Harabasz Score'])],
        'Davies-Bouldin': list(cluster_range)[np.argmin(methods['Davies-Bouldin Score'])]
    }

    return {
        'methods': methods,
        'optimal_clusters': optimal_clusters,
        'linkage_matrix': linkage_matrix
    }


if __name__ == '__main__':
    run_RSM_x_RSM()
