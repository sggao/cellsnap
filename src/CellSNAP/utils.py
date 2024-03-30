import warnings
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
import torch
import random
import torch.optim as optim
from scipy.stats import entropy



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


def get_neighborhood_composition(knn_indices, labels, full_labels = None):
    """
    Compute the composition of neighbors for each sample.
    Parameters
    ----------
    knn_indices: np.ndarray of shape (n_samples, n_neighbors)
        Each row represents the knn of that sample
    labels: np.ndarray of shape (n_samples, )
        Cluster labels
    full_labels: np.ndarray of shape (n_total_samples, )
        Cluster labels for all field of views combined

    Returns
    -------
    comp: np.ndarray of shape (n_samples, n_neighbors)
        The composition (in proportion) of neighbors for each sample.
    """

    labels = list(labels)
    n, k = knn_indices.shape
    if full_labels is not  None:
        unique_clusters = np.sort(np.unique(full_labels))
    else:
        unique_clusters = np.sort(np.unique(labels))
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
            curr_u, curr_s, curr_vh = randomized_svd(arr,
                                                     n_components=n_components,
                                                     random_state=None)
            curr_err = np.sum((arr - curr_u @ np.diag(curr_s) @ curr_vh)**2)
            if curr_err < best_err:
                best_err = curr_err
                u, s, vh = curr_u, curr_s, curr_vh
        assert u is not None and s is not None and vh is not None
    else:
        if n_runs > 1:
            warnings.warn("Doing deterministic SVD, n_runs reset to one.")
        u, s, vh = svds(arr * 1.0,
                        k=n_components)  # svds can not handle integer values
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
    u, s, vh = robust_svd(arr,
                          n_components=n_components,
                          randomized=randomized,
                          n_runs=n_runs)
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
    u, s, vh = robust_svd(arr,
                          n_components=n_components,
                          randomized=randomized,
                          n_runs=n_runs)
    return u @ np.diag(s)


def get_optimizer_and_scheduler(parameters,
                                OptimizerAlg='Adam',
                                optimizer_kwargs=None,
                                SchedulerAlg='StepLR',
                                scheduler_kwargs=None):
    if SchedulerAlg == "StepLR":
        SchedulerAlg = optim.lr_scheduler.StepLR
    elif SchedulerAlg == "MultiStepLR":
        SchedulerAlg = optim.lr_scheduler.MultiStepLR
    else:
        SchedulerAlg = None

    if OptimizerAlg == "SGD":
        OptimizerAlg = optim.SGD
    elif OptimizerAlg == "Adadelta":
        OptimizerAlg = optim.Adadelta
    elif OptimizerAlg == "Adam":
        OptimizerAlg = optim.Adam
    elif OptimizerAlg == "RMSprop":
        OptimizerAlg = optim.RMSprop
    else:
        raise NotImplementedError

    if optimizer_kwargs:
        optimizer = OptimizerAlg(parameters, **optimizer_kwargs)
    else:
        optimizer = OptimizerAlg(parameters)

    if scheduler_kwargs:
        scheduler = SchedulerAlg(optimizer, **scheduler_kwargs)
    else:
        if SchedulerAlg:
            scheduler = SchedulerAlg(optimizer)
        else:
            scheduler = None

    return optimizer, scheduler

def cluster_refine(label, label_ref, entropy_threshold = 0.75, concen_threshold = 1, max_breaks = 3):
    label_out = label.copy()
    label_out.name = label_out.name + '-refined'
    label_out = label_out.astype(str)
    ll = np.unique(label)
    for l in ll:
        ref_l = label_ref[label == l]
        ref_l_freq = ref_l.value_counts()
        if entropy(ref_l_freq) > entropy_threshold:
            for i in np.arange(max_breaks-1):
                bb = label[label_ref == ref_l_freq.index[i]]
                if entropy(bb.value_counts()) < concen_threshold:
                    label_out[(label == l) & (label_ref == ref_l_freq.index[i])] = l + '-' + str(i)
    
    return label_out.astype('category')

def clean_cluster(label):
    ll = label.value_counts().index.to_list()
    i = 0
    dd = {}
    for l in ll:
        dd[l] = str(i)
        i = i+1
    res = []
    for item in label:
        t = dd[item]
        res.append(t)
    return res