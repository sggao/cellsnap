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

#############################
# example code in this file #
#############################


def main():
    # pipeline for codex murine dataset
    df = pd.read_csv('data/codex_murine/features_and_metadata.csv',
                     index_col=0)
    features_list = [
        'CD45', 'Ly6C', 'TCR', 'Ly6G', 'CD19', 'CD169', 'CD106', 'CD3',
        'CD1632', 'CD8a', 'CD90', 'F480', 'CD11c', 'Ter119', 'CD11b', 'IgD',
        'CD27', 'CD5', 'CD79b', 'CD71', 'CD31', 'CD4', 'IgM', 'B220', 'ERTR7',
        'MHCII', 'CD35', 'CD2135', 'CD44', 'nucl', 'NKp46'
    ]
    murine_dataset = SNAP_Dataset(
        df,
        features_list=features_list,
        nbhd_composition=15,
        feature_neighbor=15,
        spatial_neighbor=15,
        path2img='../../data/tutorial/codex_murine/processed_images')
    # prepare meta data
    murine_dataset.initialize(cent_x="centroid_x",
                              cent_y="centroid_y",
                              celltype="feature_labels",
                              pca_components=25,
                              cluster_res=1.0)

    # train CellSNAP
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    murine_cellsnap = CellSNAP(murine_dataset,
                               device,
                               cnn_model=True,
                               cnn_latent_dim=128,
                               gnn_latent_dim=32)
    # Load pretrained SNAP-CNN embedding
    murine_cellsnap.cnn_embedding = np.load(
        'data/codex_murine/results/SNAP_CNN_embedding.npy')
    murine_cellsnap.get_snap_embedding(round=3,
                                       k=32,
                                       learning_rate=1e-3,
                                       n_epochs=5000,
                                       loss_fn='MSELoss',
                                       OptimizerAlg='Adam',
                                       optimizer_kwargs={},
                                       SchedulerAlg=None,
                                       scheduler_kwargs={},
                                       verbose=True)
    # clustering and visualization
    murine_cellsnap.get_snap_clustering(neighbor=15, resolution=1.0)
    murine_cellsnap.visualize_umap(murine_cellsnap.snap_embedding,
                                   murine_cellsnap.snap_clustering)


if __name__ == '__main__':
    main()
