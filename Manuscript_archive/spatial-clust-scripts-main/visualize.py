import numpy as np
import scipy


def recode(labels):
    """
    Recode labels into integers starting from zero.
    """
    unique_labels = np.unique(labels)
    old_to_new = {old: new for new, old in enumerate(unique_labels)}
    new_to_old = {new: old for new, old in enumerate(unique_labels)}
    new_labels = []
    for l in labels:
        new_labels.append(old_to_new[l])
    return np.array(new_labels), new_to_old


def fill_clusters_one_block(cell_indices, clust_labels,
                            view, path='../../data/codex_murine/segmentation_results/'):
    """
    Load the segmentation matrix for view from path,
    change the original segmentation id to its clustering label,
    then return the matrix.
    clust_labels must be coded in integers starting from zero.
    """
    islands = scipy.io.loadmat(
        '{}point{}/nuclCD45_1.6_mpp0.8/segmentationParams.mat'.format(path, view)
    )['newLmod']
    # add label by one since we reserve 0 for empty slots in segmentation
    cell_idx_to_clust_label = {idx: label + 1 for idx, label in zip(cell_indices, clust_labels)}
    # fill in cluster labels
    res = np.empty_like(islands)
    for i in range(islands.shape[0]):
        for j in range(islands.shape[1]):
            # return 0 (empty) if this cell is a dirty cell
            res[i, j] = cell_idx_to_clust_label.get(islands[i, j], 0)
    return res


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




###### added specifically for tonsil



def fill_clusters_one_block(cell_indices, clust_labels,
                            view, path='../../data/codex_murine/segmentation_results/'):
    """
    Load the segmentation matrix for view from path,
    change the original segmentation id to its clustering label,
    then return the matrix.
    clust_labels must be coded in integers starting from zero.
    """
    islands = scipy.io.loadmat(
        '{}point{}/nuclCD45_1.6_mpp0.8/segmentationParams.mat'.format(path, view)
    )['newLmod']
    # add label by one since we reserve 0 for empty slots in segmentation
    cell_idx_to_clust_label = {idx: label + 1 for idx, label in zip(cell_indices, clust_labels)}
    # fill in cluster labels
    res = np.empty_like(islands)
    for i in range(islands.shape[0]):
        for j in range(islands.shape[1]):
            # return 0 (empty) if this cell is a dirty cell
            res[i, j] = cell_idx_to_clust_label.get(islands[i, j], 0)
    return res


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

