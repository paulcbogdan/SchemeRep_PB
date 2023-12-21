import os
import pathlib
from collections import defaultdict

import numpy as np
import scipy.stats as stats
from tqdm import tqdm



def get_NBS_clusters(p_graph, alpha_thresh):
    # t_cutoff = stats.t.ppf(1 - alpha_thresh, N - 1)  # one-sided p-value
    # top_edges = np.argwhere(abs(t_graph) > t_cutoff)
    top_edges = np.argwhere(p_graph < alpha_thresh)
    root2leafs = defaultdict(list)
    if len(top_edges) == 0:
        return root2leafs, 0
    for edge in top_edges:
        root2leafs[edge[0]].append(edge[1])

    already_done_nodes = set()
    components = []  # list of lists of tuples,
    # e.g., components[0] = [(0, 1), (0, 2)] means the 0th
    #   component contains (ROI0, ROI1) and (ROI0, ROI2) edges
    biggest_size = 0
    for i in tqdm(root2leafs):
        if i not in already_done_nodes:
            # Use recursion in process_node(...) to find clusters
            component_nodes, component_edges, _ = process_node(i, root2leafs)
            assert len(component_edges) == len(set(component_edges)), \
                'Programming error'  # Sanity check. This should never trigger.

            # omit symmetric duplicates - e.g., keep only (0, 1) and not (1, 0))
            component_edges = [edge for edge in component_edges if
                               edge[0] < edge[1]]
            components.append(component_edges)

            # Future iterations will skip over any nodes in present cluster
            already_done_nodes.update(component_nodes)

            biggest_size = max(biggest_size, len(component_edges))
    components.sort(key=lambda x: len(x), reverse=True)
    return components, biggest_size


def process_node(i, root2leafs, already_done=None):
    '''
    Recursively find all nodes connected to node i. Depth-first search.
    :param i: int, ROI index
    :param root2leafs: dict, keys are ROI indices, values are lists of indices
                             indicating ROIs connected to the key ROI
    :param already_done: set, indices of ROIs already processed
    :return: i_connected_nodes, list of ROIs in component
             i_connected_edges, list of edges in component
             already_done, set of ROIs already processed (for recursion)
    '''
    if already_done is None:
        already_done = set()
    already_done.add(i)
    i_connected_nodes = root2leafs[i]
    i_connected_edges = [(i, j) for j in root2leafs[i] if i != j]
    for j in root2leafs[i]:
        if j in root2leafs and j not in already_done:
            j_connected_nodes, j_connected_edges, already_done_mod = \
                process_node(j, root2leafs, already_done=already_done)
            i_connected_nodes.extend(j_connected_nodes)
            i_connected_edges.extend(j_connected_edges)
            already_done.update(already_done)
    return i_connected_nodes, i_connected_edges, already_done
