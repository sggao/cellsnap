import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import utils
import graph
from preprocessing import *


class SNAP_Dataset(Dataset):
    # SNAP Dataset Object

    def __init__(self,
                 df,
                 k=15,
                 feature_neighbor=15,
                 spatial_neighbor=15, 
                 pca_components=25,
                 fov_list=None,
                 features_list=None,
                 path2img=None,
                 use_transform=False):
        """
        Form dataset of spatial single-cell data
        Parameters
        ----------

        df : pandas dataframe
            dataframe containing meta information: require columns 'centroid_x', 'centroid_y'
            also require column 'fov', indicating indices of the field of views (if contains more than 1 fov)
        k : int 
            number of neighboring cells in neighborhood composition vector, default 15
        feature_neighbor : int
            number of neighbors in SNAP GNN
        pca_component : int
            number of PCs in feature graph
        features_list : list(str)
            list of feature names to be extracted from df
        path2img : str, optional
            path to images (to be saved)
        use_transform : bool
            indicate whether to use data augmentation

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
        self.spatial_neighbor = spatial_neighbor
        self.pca_components = pca_components
        self.features_list = features_list
        self.path2img = path2img
        self.use_transform = use_transform
        self.fov_list = fov_list if fov_list else [0]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = np.load(os.path.join(self.path2img, f"img_{index:05d}.npy"))
        if self.use_transform:
            img = self.transform(torch.Tensor(img))
        labels = self.labels[index]
        return img, labels

    def initialize(self,
                   cent_x,
                   cent_y,
                   celltype,
                   resolution=1.0,
                   cluster=None,
                   n_runs=1,
                   resolution_tol=0.05):
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
            arr=self.features,
            pca_components=self.pca_components,
            n_neighbors=self.feature_neighbor,
            metric='correlation',
            verbose=False)
        self.feature_labels = graph.graph_clustering(
            self.df.shape[0],
            self.feature_edges,
            resolution=resolution,
            n_clusters=cluster,
            n_runs=n_runs,
            resolution_tol=resolution_tol,
            seed=None,
            verbose=False)
        self.df['feature_labels'] = self.feature_labels

        print('Calculating cell neighborhood composition matrix...')
        self.locations = []

        locations = self.df[[cent_x, cent_y]].to_numpy()
        self.locations.append(locations)
        spatial_knn_indices = graph.get_spatial_knn_indices(
            locations=locations, n_neighbors=self.k, method='kd_tree')
        cell_nbhd = utils.get_neighborhood_composition(
            knn_indices=spatial_knn_indices,
            labels=self.df[celltype],
            full_labels=self.df[celltype])
        
        self.labels = cell_nbhd
        # create dual label for GNN
        cell_label = pd.get_dummies(self.feature_labels)
        dual_label = np.hstack([cell_label, self.labels])
        self.dual_labels = dual_label
        
        # create spatial edges
        self.spatial_edges = graph.get_spatial_edges(
                                arr=locations, n_neighbors=self.spatial_neighbor, verbose=True
                            )

    def prepare_images(self,
                       image,
                       size,
                       truncation,
                       aggr,
                       pad=1000,
                       verbose=False):

        n_cells = self.df.shape[0]
        power = len(str(n_cells))
        print('Saving images...')
        process_save_images(images=image,
                            locations=self.locations,
                            size=size,
                            fov_list=self.fov_list,
                            save_folder=self.path2img,
                            truncation=truncation,
                            power=power, 
                            aggr=aggr,
                            pad=1000,
                            verbose=False)
