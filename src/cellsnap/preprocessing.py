import numpy as np
import pandas as pd
import scipy
from skimage.measure import regionprops
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_save_images(images,
                        locations,
                        size,
                        save_folder,
                        truncation,
                        power=5,
                        aggr=[[0], [1]],
                        pad=1000,
                        verbose=False):
    
    """
    Helper function to produce cropped images for each individual cells and same them out.
    The save out images will be used in the SNAP-CNN process to extract morphology encoding.
    Parameters
    ----------
    image: np.array with (H,W,2)
        One tissue image file in the format of numpy array. Note the supplied x, y location of cells from
        the initial df should be the same as their pixel location in this supplied image. Two channels
        corresponds to the membrane and nuclear channels.
    size: int
        Size of the cropped individual images for each cell.
    truncation: float
        Quantile value as threshold to binarize the input image.
    aggr: list
        Default not used
    pad: int
        Padding value around input image.
    """

    img_idx = 0
    image = images
    pad_image = np.zeros(
        (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad, image.shape[2]))
    pad_image[pad:image.shape[0] + pad, pad:image.shape[1] + pad, :] = image
    truncate = np.quantile(pad_image, q=truncation, axis=(0, 1))
    truncate = truncate[None, None, :]

    pad_image[pad_image <= truncate] = 0
    pad_image[pad_image > truncate] = 1

    pad_image_sum = np.zeros([pad_image.shape[0], pad_image.shape[1], 2])
    pad_image_sum[:, :, 0] = np.sum(pad_image[:, :, i] for i in aggr[0])
    pad_image_sum[:, :, 1] = np.sum(pad_image[:, :, i] for i in aggr[1])

    sub_location = locations
    n_cells = sub_location.shape[0]
    power = len(str(n_cells))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if verbose:
        print("Processing each cell...and saving!", flush=True)
    for i in tqdm(range(n_cells)):
        # process each cell
        center_x = sub_location[i][0]
        center_y = sub_location[i][1]
        cur_image = np.transpose(
            pad_image_sum[(int(center_x) - size // 2 + pad):(int(center_x) +
                                                             size // 2 + pad),
                          (int(center_y) - size // 2 +
                           pad):(int(center_y) + size // 2 + pad), :],
            (2, 0, 1)).astype(np.int8)
        assert (cur_image.shape == (2, size, size))
        if verbose:
            if i % 10000 == 1:
                plt.imshow(cur_image[0, :, :])
                plt.show()
                plt.imshow(cur_image[1, :, :])
                plt.show()

        np.save(file=os.path.join(save_folder, f"img_{i:0{power}d}"),
                arr=cur_image)
        img_idx += 1

    return


########################################################################################
#         Below is legacy functions not used in the actual CellSNAP pipeline.          #
########################################################################################

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


def add_cell_locations(df,
                       path_to_segmentation,
                       shape_of_views=(9, 7),
                       shape_of_each_view=(1008, 1344),
                       verbose=True):
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
            topleft = [
                view_i * shape_of_each_view[0], view_j * shape_of_each_view[1]
            ]
            if verbose:
                print("Now at field of view {}, top-left coordinate is {}".
                      format(view, topleft),
                      flush=True)
            seg = scipy.io.loadmat(
                '{}point{}/nuclCD45_1.6_mpp0.8/segmentationParams.mat'.format(
                    path_to_segmentation, view))['newLmod']
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
                seg_label_to_x[unique_seg_labels[
                    i]] = props[i]['centroid'][0] + topleft[0]
                seg_label_to_y[unique_seg_labels[
                    i]] = props[i]['centroid'][1] + topleft[1]
            # fill the centroids of this segment of df
            start, end = partition[view - 1]
            for i in range(start, end + 1):
                centroid_x.append(
                    seg_label_to_x[df.iloc[i]['cellLabelInImage']])
                centroid_y.append(
                    seg_label_to_y[df.iloc[i]['cellLabelInImage']])
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
        'CD45', 'Ly6C', 'TCR', 'Ly6G', 'CD19', 'CD169', 'CD106', 'CD3',
        'CD1632', 'CD8a', 'CD90', 'F480', 'CD11c', 'Ter119', 'CD11b', 'IgD',
        'CD27', 'CD5', 'CD79b', 'CD71', 'CD31', 'CD4', 'IgM', 'B220', 'ERTR7',
        'MHCII', 'CD35', 'CD2135', 'CD44', 'nucl', 'NKp46'
    ]
    feature_to_idx = {f: i for i, f in enumerate(features)}
    indices = [feature_to_idx[f] for f in channels]
    img = np.concatenate([
        img[0, :, :, :],
        np.array([img[i, :, :, j] for i in range(1, 14)
                  for j in range(1, 3)]).transpose((1, 2, 0)),
        img[16, :, :, 2][:, :, np.newaxis], img[17, :, :, 1][:, :, np.newaxis]
    ],
                         axis=2)
    return img[:, :, indices]