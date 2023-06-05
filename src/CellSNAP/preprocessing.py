import numpy as np
import pandas as pd
import scipy
from skimage.measure import regionprops
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize


def get_cell_idx_partition(df):
    """
    Return partition s.t. partition[i] = [start_i, end_i] are the
    starting and ending indices (closed interval) in df for the i-th FOV (indexing from 1).
    """
    starts = [0]
    for i, curr_cell_idx in enumerate(df['cellLabelInImage']):
        if i > 0 and curr_cell_idx < df['cellLabelInImage'].iloc[i - 1]:
            starts.append(i)
    partition = []
    for i, s in enumerate(starts):
        if i < len(starts) - 1:
            partition.append([s, starts[i + 1] - 1])
        else:
            partition.append([s, df.shape[0] - 1])
    return partition


def add_cell_locations(
        df, path_to_segmentation, shape_of_views=(9, 7), shape_of_each_view=(1008, 1344), verbose=True
):
    """
    Add three new columns to df: location coordinates (x, y) as well as which FOV is each cell in.
    """
    # get the splitting points of different FOVs
    partition = get_cell_idx_partition(df)
    centroid_x = []
    centroid_y = []
    cell_views = []
    for view_j in range(shape_of_views[1]):
        for view_i in range(shape_of_views[0]):
            view = view_j * shape_of_views[0] + view_i + 1
            topleft = [view_i * shape_of_each_view[0], view_j * shape_of_each_view[1]]
            if verbose:
                print("Now at field of view {}, top-left coordinate is {}".format(view, topleft), flush=True)
            seg = scipy.io.loadmat(
                '{}point{}/nuclCD45_1.6_mpp0.8/segmentationParams.mat'.format(path_to_segmentation, view)
            )['newLmod']
            # get unique labels, excluding zero
            unique_seg_labels = list(np.unique(seg.flatten()))[1:]
            # calculate centroids
            props = regionprops(seg)
            # unique labels should align with props
            assert len(unique_seg_labels) == len(props)
            # build dict of seg_label: x and seg_label: y
            seg_label_to_x = {}
            seg_label_to_y = {}
            for i in range(len(props)):
                seg_label_to_x[unique_seg_labels[i]] = props[i]['centroid'][0] + topleft[0]
                seg_label_to_y[unique_seg_labels[i]] = props[i]['centroid'][1] + topleft[1]
            # fill the centroids of this segment of df
            start, end = partition[view - 1]
            for i in range(start, end + 1):
                centroid_x.append(seg_label_to_x[df.iloc[i]['cellLabelInImage']])
                centroid_y.append(seg_label_to_y[df.iloc[i]['cellLabelInImage']])
                cell_views.append(view)
    # add new columns
    df['centroid_x'] = centroid_x
    df['centroid_y'] = centroid_y
    df['field_of_view'] = cell_views


def select_useful_features(img, channels=('CD45', 'nucl')):
    """
    Select useful channels in the whole tiff image of shape (18, 1008, 1344, 3),
    and get an np array of shape (1008, 1344, 31).
    Finally return an np array of shape (1008, 1344, len(channels))
    """
    features = [
        'CD45',
        'Ly6C',
        'TCR',
        'Ly6G',
        'CD19',
        'CD169',
        'CD106',
        'CD3',
        'CD1632',
        'CD8a',
        'CD90',
        'F480',
        'CD11c',
        'Ter119',
        'CD11b',
        'IgD',
        'CD27',
        'CD5',
        'CD79b',
        'CD71',
        'CD31',
        'CD4',
        'IgM',
        'B220',
        'ERTR7',
        'MHCII',
        'CD35',
        'CD2135',
        'CD44',
        'nucl',
        'NKp46'
    ]
    feature_to_idx = {f: i for i, f in enumerate(features)}
    indices = [feature_to_idx[f] for f in channels]
    img = np.concatenate(
        [img[0, :, :, :],
         np.array([img[i, :, :, j] for i in range(1, 14) for j in range(1, 3)]).transpose((1, 2, 0)),
         img[16, :, :, 2][:, :, np.newaxis],
         img[17, :, :, 1][:, :, np.newaxis]
        ],
        axis=2
    )
    return img[:, :, indices]