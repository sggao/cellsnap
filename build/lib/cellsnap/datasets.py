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
                 nbhd_composition=15,
                 feature_neighbor=15,
                 spatial_neighbor=15,
                 features_list=None,
                 path2img=None,
                 use_transform=False):
        """
        Form dataset of spatial single-cell data
        Parameters
        ----------

        df : pandas dataframe containing information:
            - feature expression values (eg protein or RNA values etc)
            - 2D cell location information (x,y):
                @ cell location need to be global in the case of multiple seperated FOV (eg TMAs) to avoid incorrect spatial
                adjacency and neighborhood information.
                @ if will be training SNAP-CNN (extracting morphology information from image files), the supplied 2D cell
                location inforamtion should be consistent with the pixel locations of cell location in the input image files. 
        nbhd_composition : int 
            Number of neighboring cells in neighborhood composition vector, default 15
        feature_neighbor : int
            Number of neighbors to consider in feature similarity graph
        spatial_neighbor : int
            Number of neighbors to consider in spatial similarity graph
        features_list : list(str)
            List of feature names to be used in CellSNAP process
        path2img : str, optional
            Path to images (to be saved, output from function '.prepare_images') if using images to extract morphological information (CNN step)
        use_transform : bool
            Indicate whether to use data augmentation if using images to extract morphological information (CNN step)

        """

        super().__init__()
        self.df = df
        self.nbhd_composition = nbhd_composition
        self.feature_neighbor = feature_neighbor
        self.spatial_neighbor = spatial_neighbor
        self.features_list = features_list
        self.path2img = path2img
        self.use_transform = use_transform
        if self.use_transform:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img = np.load(
            os.path.join(self.path2img, f"img_{index:0{self.power}d}.npy"))
        if self.use_transform:
            img = self.transform(torch.Tensor(img))
        labels = self.labels[index]
        return img, labels

    def initialize(self,
                   cent_x,
                   cent_y,
                   celltype='feature_labels',
                   pca_components=25,
                   cluster_res=1.0,
                   resolution_tol=0.05,
                   n_clusters=None,
                   n_runs=1):
        """
        Initialize the class object by calculating:
        -feature edges from similarity, used in SNAP-GNN
        -feature labels from feature values (population identity), default by Leiden clustering (skip if supplied)
        -neighborhood composition
        -spatial adjacency related information
        Parameters
        ----------

        cent_x, cent_y : str
            Column names of self.df, which contains location of each cell
        celltype : str
            Defualt to 'feature_labels' which will run leiden clustering clustering 
            and produce population identity of each cell. Otherwise set to a column in self.df 
            that contains the initial cell type annotation information
        pca_component : int
            Number of PCs used in expression graph. if 'None' no PCA reduction will be performed on the input feature profile
        cluster_res, resolution_tol : float, float
            Resolution parameter for leiden clustering on features
        cluster : int
            Leiden clustering parameter, fix the number of output labels
        n_runs : int
            Number of rounds for computing leiden clustering, default to be 1

        """

        # create metadata for the dataset
        self.n_cells = self.df.shape[0]
        self.power = len(str(self.n_cells))
        features = self.df[self.features_list].to_numpy(
        ) if self.features_list is not None else self.df.to_numpy()
        self.features = utils.center_scale(features)
        self.cluster_res = cluster_res
        self.resolution_tol = resolution_tol
        self.pca_components = pca_components

        # create feature edges and feature labels
        self.feature_edges = graph.get_feature_edges(
            arr=self.features,
            pca_components=self.pca_components,
            n_neighbors=self.feature_neighbor,
            metric='correlation',
            verbose=False)
        if celltype == "feature_labels":
            self.feature_labels = graph.graph_clustering(
                self.df.shape[0],
                self.feature_edges,
                resolution=self.cluster_res,
                n_clusters=n_clusters,
                n_runs=n_runs,
                resolution_tol=self.resolution_tol,
                seed=None,
                verbose=False)
            self.df['feature_labels'] = self.feature_labels

            print('Leiden clustering identified ' +
                  str(len(np.unique(self.feature_labels))) +
                  ' clusters as input population identity.')
        else:
            print('Predefined annotation identified ' +
                  str(len(np.unique(self.df[celltype]))) +
                  ' clusters as input population identity.')

        print('Calculating cell neighborhood composition matrix...')

        locations = self.df[[cent_x, cent_y]].to_numpy()
        self.locations = locations
        spatial_knn_indices = graph.get_spatial_knn_indices(
            locations=locations,
            n_neighbors=self.nbhd_composition,
            method='kd_tree')
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
            arr=locations, n_neighbors=self.spatial_neighbor, verbose=True)

    def prepare_images(self,
                       image,
                       size,
                       truncation,
                       aggr,
                       pad=1000,
                       verbose=False):
        """
        Helper function takes input of a whole tissue iamge with nuclear and membrane channel.
        Output individual cropped binarized images for each cell of their adjacent tissue information.
        The images will be saved in the pre-specified saving directory in function '.SNAP_Dataset'.
        Parameters
        ----------

        image: np.array with (H,W,2)
            One tissue image file in the format of numpy array. Note the supplied x, y location of cells from
            the initial df should be the same as their pixel location in this supplied image. two channels
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

        print('Saving images...')
        process_save_images(images=image,
                            locations=self.locations,
                            size=size,
                            save_folder=self.path2img,
                            truncation=truncation,
                            power=self.power,
                            aggr=aggr,
                            pad=pad,
                            verbose=verbose)
