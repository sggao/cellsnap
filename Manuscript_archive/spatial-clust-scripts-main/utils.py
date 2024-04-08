import warnings
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
import torch
import random


def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)


def center_scale(arr):
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def drop_zero_variability_columns(arr, tol=1e-8):
    """
    Drop columns for which its standard deviation is zero in any one of the arrays in arr_list.
    Parameters
    ----------
    arr: np.ndarray of shape (n_samples, n_features)
        Data matrix
    tol: float, default=1e-8
        Any number less than tol is considered as zero
    Returns
    -------
    np.ndarray where no column has zero standard deviation
    """
    bad_columns = set()
    curr_std = np.std(arr, axis=0)
    for col in np.nonzero(np.abs(curr_std) < tol)[0]:
        bad_columns.add(col)
    good_columns = [i for i in range(arr.shape[1]) if i not in bad_columns]
    return arr[:, good_columns]


def get_neighborhood_composition(knn_indices, labels):
    """
    Compute the composition of neighbors for each sample.
    Parameters
    ----------
    knn_indices: np.ndarray of shape (n_samples, n_neighbors)
        Each row represents the knn of that sample
    labels: np.ndarray of shape (n_samples, )
        Cluster labels

    Returns
    -------
    comp: np.ndarray of shape (n_samples, n_neighbors)
        The composition (in proportion) of neighbors for each sample.
    """
    labels = list(labels)
    n, k = knn_indices.shape
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    label_to_clust_idx = {label: i for i, label in enumerate(unique_clusters)}

    comp = np.zeros((n, n_clusters))
    for i, neighbors in enumerate(knn_indices):
        good_neighbors = [nb for nb in neighbors if nb != -1]
        for nb in good_neighbors:
            comp[i, label_to_clust_idx[labels[nb]]] += 1

    return (comp.T / comp.sum(axis=1)).T


def robust_svd(arr, n_components, randomized=False, n_runs=1):
    """
    Do deterministic or randomized SVD on arr.
    Parameters
    ----------
    arr: np.array
        The array to do SVD on
    n_components: int
        Number of SVD components
    randomized: bool, default=False
        Whether to run randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    u, s, vh: np.array
        u @ np.diag(s) @ vh is the reconstruction of the original arr
    """
    if randomized:
        best_err = float('inf')
        u, s, vh = None, None, None
        for _ in range(n_runs):
            curr_u, curr_s, curr_vh = randomized_svd(arr, n_components=n_components, random_state=None)
            curr_err = np.sum((arr - curr_u @ np.diag(curr_s) @ curr_vh) ** 2)
            if curr_err < best_err:
                best_err = curr_err
                u, s, vh = curr_u, curr_s, curr_vh
        assert u is not None and s is not None and vh is not None
    else:
        if n_runs > 1:
            warnings.warn("Doing deterministic SVD, n_runs reset to one.")
        u, s, vh = svds(arr*1.0, k=n_components) # svds can not handle integer values
    return u, s, vh


def svd_denoise(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute best rank-n_components approximation of arr by SVD.
    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    arr: array_like of shape (n_samples, n_features)
        Rank-n_comopnents approximation of the input arr.
    """
    if n_components is None:
        return arr
    u, s, vh = robust_svd(arr, n_components=n_components, randomized=randomized, n_runs=n_runs)
    return u @ np.diag(s) @ vh


def svd_embedding(arr, n_components=20, randomized=False, n_runs=1):
    """
    Compute rank-n_components SVD embeddings of arr.
    Parameters
    ----------
    arr: np.array of shape (n_samples, n_features)
        Data matrix
    n_components: int, default=20
        Number of components to keep
    randomized: bool, default=False
        Whether to use randomized SVD
    n_runs: int, default=1
        Run multiple times and take the realization with the lowest Frobenious reconstruction error
    Returns
    -------
    embeddings: array_like of shape (n_samples, n_components)
        Rank-n_comopnents SVD embedding of arr.
    """
    if n_components is None:
        return arr
    u, s, vh = robust_svd(arr, n_components=n_components, randomized=randomized, n_runs=n_runs)
    return u @ np.diag(s)




###### added by bokai 1114-2022

def fill_clusters_one_column(df, clust_labels, views,
                             path='../../data/codex_murine/segmentation_results/',
                             colnames=('cell_view', 'cellLabelInImage')):
    """
    Fill cluster labels for cells in views,
    vertically concatenate the filled segmentation matrices, and return it.
    df must contain the following columns:
        - colnames[0]: which view is each cell in
        - colnames[1]: the segmentation index of each cell.
    clust_labels must be coded in integers starting from zero.
    """
    mask = df[colnames[0]] == views[0]
    islands = fill_clusters_one_block(
        df[mask][colnames[1]], clust_labels[mask], views[0], path
    )
    for i in range(1, len(views)):
        mask = df[colnames[0]] == views[i]
        islands = np.concatenate(
            (islands,
             fill_clusters_one_block(
                 df[mask][colnames[1]], clust_labels[mask], views[i], path
             )
             ),
            axis=0
        )
    return islands


def fill_clusters_to_segmentation(df, views, shape,
                                  path='../../data/codex_murine/segmentation_results/',
                                  colnames=('cell_view', 'cellLabelInImage', 'clust_label')):
    """
    Fill cluster labels to the segmentation matrices in views,
    concatenate them, and return the overall matrix.
    Also return idx_to_label, a dict of {idx_in_seg_mat_after_filling_in_clust_labels: original_clust_label}.
    df must contain the following columns:
        - colnames[0]: which view is each cell in
        - colnames[1]: the segmentation index of each cell
        - colnames[2]: the cluster label of each cell.
    """
    assert shape[0] * shape[1] == len(views)
    # recode clusters to integers starting from zero
    clust_labels, new_to_old = recode(df[colnames[2]])
    # in the filling process, 0 is reserved for empty
    new_to_old = {new + 1: old for new, old in new_to_old.items()}
    new_to_old[0] = 'empty'

    # fill in the first column
    start, end = 0, shape[0]
    islands = fill_clusters_one_column(df, clust_labels, views[start:end], path, colnames[:2])
    while end < len(views):
        start = end
        end += shape[0]
        islands = np.concatenate(
            (islands, fill_clusters_one_column(
                df, clust_labels, views[start:end], path, colnames[:2]
            )), axis=1
        )
    return islands, new_to_old
