import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import utils
import graph


class SNAP_Dataset(Dataset):
    # SNAP Dataset Object

    def __init__(self, df, k, feature_neighbor, pca_component, features_list, path2img):
        """
        Form dataset of spatial single-cell data
        Parameters
        ----------

        df : pandas dataframe
            dataframe containing meta information: containing 'centroid_x', 'centroid_y'
        k : int 
            number of neighboring cells in neighborhood composition vector, default 15
        feature_neighbor : int
            number of neighbors in SNAP GNN
        pca_component : int
            number of PCs in feature graph
        features_list : list(str)
            list of feature names to be extracted from df
        path2img : str, optional
            path to images

        """

        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        self.df = df
        self.k = k
        self.feature_neighbor = feature_neighbor
        self.pca_component = pca_component
        self.features_list = features_list

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = np.load(os.path.join(self.img_path, f"img_{index:05d}.npy"))
        if self.use_transform:
            img = self.transform(torch.Tensor(img))
        labels = self.labels[index]
        return img, labels

    def initialize(self, cent_x, cent_y, celltype, n_runs = 1, resolution_tol = 0.05):
        """
        Parameters
        ----------

        cent_x, cent_y : str
            columns of self.df, location of center of each cell
        celltype : str
            column of self.df, celltype column name

        """
        # create metadata for the dataset
        features = self.df[self.features_list].to_numpy()
        self.features = utils.center_scale(features)

        # create feature edges and feature labels
        self.feature_edges = graph.get_feature_edges(
            arr=features, pca_components=self.pca_components,
            n_neighbors=self.feature_neighbor, metric='correlation', verbose=False
        )
        self.feature_labels = graph.graph_clustering(
            df.shape[0], feature_edges, resolution=res, n_clusters=None, n_runs=n_runs,
            resolution_tol=resolution_tol, seed=None, verbose=False
        )
        print('Calculating cell neighborhood composition matrix...')
        locations = self.df[['centroid_x', 'centroid_y']].to_numpy()
        self.spatial_knn_indices = graph.get_spatial_knn_indices(locations=locations, 
                n_neighbors=np.unique(self.df[celltype]).shape[0], method='kd_tree')



        

        return

    def prepare_images():
        return
