import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import skimage
import sys
sys.path.append("../src/CellSNAP/")
from utils import *
import os
from tqdm import tqdm
from skimage.io import imread
from preprocessing import *
from datasets import *
from cellsnap import *


def main():
    # pipeline for codex murine dataset
    df = pd.read_csv('../data/codex_murine/features_and_metadata.csv', index_col = 0)
    features_list = [
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
    murine_dataset = SNAP_Dataset(df,
                             k=15,
                             feature_neighbor=15,
                             pca_components=25,
                             features_list=features_list,
                             path2img='../../data/tutorial/codex_murine/processed_images')

    murine_dataset.initialize("centroid_x", "centroid_y", "cluster.termV2")
    shape_of_each_view = (1008, 1344)
    shape_of_views = (9, 7)
    channels=('CD45', 'nucl')
    load_path = '../../data/codex_murine'

    image = np.zeros((shape_of_each_view[0]*shape_of_views[0], 
                            shape_of_each_view[1]*shape_of_views[1], len(channels)))
    for view_j in range(shape_of_views[1]):
        for view_i in range(shape_of_views[0]):
            view = view_j * shape_of_views[0] + view_i + 1
            img = imread(
                    '{}/raw_images/focused_BALBc-1_X0{}_Y0{}.tif'.format(
                        load_path, str(view_j + 1), str(view_i + 1)
                    )
                )
            image[view_i*shape_of_each_view[0]:(view_i+1)*shape_of_each_view[0],
                    view_j*shape_of_each_view[1]:(view_j+1)*shape_of_each_view[1],:] = select_useful_features(img, channels)

    size = 512
    truncation = 0.8
    murine_dataset.prepare_images(image,
                                size,
                                truncation,
                                pad=1000,
                                verbose=False)
    
    # train CellSNAP
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    murine_cellsnap = CellSNAP(murine_dataset,
                            device,
                            cnn_model=True,
                            cnn_latent_dim=128,
                            gnn_latent_dim=32)
    murine_cellsnap.fit_snap_cnn(
                     batch_size=64,
                     learning_rate=1e-4,
                     n_epochs=300,
                     loss_fn='MSELoss',
                     OptimizerAlg='Adam',
                     optimizer_kwargs=None,
                     SchedulerAlg=None,
                     scheduler_kwargs=None,
                     print_every=1)
    murine_cellsnap.pred_cnn_embedding(batch_size=512, path2result='../data/codex_murine/results')
    murine_cellsnap.fit_snap_gnn(learning_rate=1e-3,
                             n_epochs=5000,
                             loss_fn='MSELoss',
                             OptimizerAlg='Adam',
                             optimizer_kwargs=None,
                             SchedulerAlg=None,
                             scheduler_kwargs=None,
                             print_every=500,
                             verbose=True)
    # visualization
    murine_cellsnap.visualize_umap(murine_cellsnap.embedding, murine_dataset.df['cluster.termV2'])

if __name__ == '__main__':
    main()


