import torch
from torch.utils.data import Dataset
import sys
sys.path.append("../../../../../")
import model


import warnings
import numpy as np
import leidenalg
import igraph as ig
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import graph

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
import skimage
# import custom functions
import sys
import utils
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv


## during testing only test snap_gnn

class SNAP_GNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(in_features=args.gnn_input_dim, out_features=args.fc_dim)
        self.cnn_fc = nn.Linear(in_features=args.cnn_input_dim, out_features=args.cnn_dim)
        self.feat_conv1 = GCNConv(args.fc_dim, args.latent_dim)
        self.feat_conv2 = GCNConv(args.latent_dim, args.fc_out_dim)
        
        self.spat_conv1 = GCNConv(args.cnn_dim, args.cnn_latent_dim)
        self.spat_conv2 = GCNConv(args.cnn_latent_dim, args.cnn_out_dim)
        
        self.proj1 = nn.Linear(in_features=args.fc_out_dim+args.cnn_out_dim, 
                              out_features=args.hid_out_dim)
        self.proj2 = nn.Linear(in_features=args.hid_out_dim, 
                              out_features=args.out_dim)
        #self.proj = nn.Linear(in_features=args.fc_out_dim+args.cnn_out_dim, 
                              #out_features=args.out_dim)

    def feat_gnn_encoder(self, feat, feat_edge_index):
        feat = F.relu(self.fc(feat))
        feat = F.relu(self.feat_conv1(feat, feat_edge_index))
        feat = self.feat_conv2(feat, feat_edge_index)
        
        return feat
    
    def spat_gnn_encoder(self, spat, spat_edge_index):
        spat = F.relu(self.cnn_fc(spat))
        spat = F.relu(self.spat_conv1(spat, spat_edge_index))
        spat = self.spat_conv2(spat, spat_edge_index)
        
        return spat
    
    def encoder(self, feat, spat, feat_edge_index, spat_edge_index):
        x_feat = self.feat_gnn_encoder(feat, feat_edge_index)
        x_spat = self.spat_gnn_encoder(spat, spat_edge_index)
        x = torch.cat((x_feat, x_spat), dim = 1)
        return x
    

    def forward(self, feat, spat, feat_edge_index, spat_edge_index):
        x = F.relu(self.encoder(feat, spat, feat_edge_index, spat_edge_index))
        x = self.proj1(x)
        x = F.relu(x)
        x = self.proj2(x)
        return x

    
    
class Args:
    gnn_input_dim = 31
    cnn_input_dim = 128
    fc_dim = latent_dim = 32
    cnn_dim = cnn_latent_dim = 32
    out_dim = 15 * 2
    #fc_out_dim = cnn_out_dim = 16
    fc_out_dim = 33
    cnn_out_dim = 11
    hid_out_dim = 33

    criterion = "L1"
    learning_rate = 1e-3
    epochs = 10000
    print_every = 500
    average_iter = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

args = Args()

def train_gnn_get_testloss(features = None, cnn_feature = None, feat_edge_index = None,
                           spat_edge_index = None, cell_nbhd = None, train_mask = None, model = None, 
                           args = None, verbose = False): # cell_nbhd not used change to combo_nbhd
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_nbhd = cell_nbhd[train_mask].to(args.device)
    test_nbhd = cell_nbhd[~train_mask].to(args.device)
    model = model.to(args.device)
    if args.criterion == "L1":
        print("Use L1 Loss")
        criterion = nn.L1Loss()
    elif args.criterion == "L2":
        print("Use L2 Loss")
        criterion = nn.MSELoss()
    else:
        print("Cross Entropy")
        criterion = nn.CrossEntropyLoss()
    
    train_loss_epoch = []
    test_loss_epoch = []
    #criterion = nn.L1Loss()
    for e in range(1, 1+args.epochs):
        model.train()
        if cnn_feature != None:
            predicted_nbhd = model(features, cnn_feature, feat_edge_index, spat_edge_index)
        else:
            predicted_nbhd = model(x=features,  edge_index=edge_index) # actually not used this line
        
        # Compute prediction error
        loss = criterion(predicted_nbhd[train_mask], train_nbhd)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # take one step
        optimizer.step()

        # record the loss
        curr_train_loss = loss.item()
        if verbose and e % args.print_every  == 0:
            print(f'===Epoch {e}, the training loss is {curr_train_loss:>0.8f}==', flush=True)
        train_loss_epoch.append(curr_train_loss)

        model.eval()
        with torch.no_grad():
            if cnn_feature != None:
                predicted_nbhd = model(features, cnn_feature, feat_edge_index, spat_edge_index)
            else:
                predicted_nbhd = model(x=features, edge_index=edge_index) # again not used here
            loss = criterion(predicted_nbhd[~train_mask], test_nbhd)
            curr_test_loss = loss.item()
            if verbose and e % args.print_every == 0:
                print(f'===Epoch {e}, the test loss is {curr_test_loss:>0.8f}===', flush=True)
            test_loss_epoch.append(curr_test_loss)
    #return test_loss_epoch
    return  np.mean(test_loss_epoch[-args.average_iter:])


l = 1
d = defaultdict(list)
# this part test truncation

for i in tqdm(range(5)):
    for a in [256, 512, 1024]:
        
        metaload_path = '/mnt/cloud1/sheng-projects/st_projects/spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/benchmark/spleen/data/'
        
        df_clean = pd.read_csv('/mnt/cloud1/sheng-projects/st_projects/spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/spleen/data/features_and_metadata.csv', index_col=0)
        features = np.load('/mnt/cloud1/sheng-projects/st_projects/spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/spleen/data/feature_scaled.npy')
        
        cell_nbhd = np.load(os.path.join(metaload_path,  f"cell_nbhd_res0.5_k20.npy"))
        train_mask = np.load(os.path.join(metaload_path,  "train_mask.npy"))
        feature_labels = np.load(os.path.join(metaload_path,  f"feature_labels_res0.5.npy"))
        feature_edges = np.load(os.path.join(metaload_path,  f"feature_edges_res0.5.npy"))
        spatial_edges = np.load(os.path.join(metaload_path,  "spatial_knn_indices_k15.npy"))                       
                               
        # change into torch
        features = torch.from_numpy(features).float().to(args.device)
        feat_edge_index = torch.from_numpy(np.array(feature_edges.T[:2])).long().to(args.device)
        spat_edge_index = torch.from_numpy(np.array(spatial_edges.T[:2])).long().to(args.device)
        
        # combo nbhd                       
        df_clean['res'] = feature_labels
        reslabel = pd.get_dummies(df_clean['res'])
        combo_nbhd = np.hstack([reslabel, cell_nbhd])
        combo_nbhd = torch.from_numpy(combo_nbhd).float().to(args.device)
        
        ## cnn
        load_path = '/mnt/cloud1/sheng-projects/st_projects/spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/benchmark/spleen/data/'
        save_folder = os.path.join(load_path, "cnn", f"cnn_{a}_l{l}_layer6_testimgsize:{a}_checkpoints", "epochs", 'embed')
        cri = ["L1", "L2", "CE"]
        args.out_dim = combo_nbhd.shape[1]
        
        #### reset args dead
        class Args:
            gnn_input_dim = 31
            cnn_input_dim = 128
            fc_dim = latent_dim = 32
            cnn_dim = cnn_latent_dim = 32
            out_dim = combo_nbhd.shape[1]
            #fc_out_dim = cnn_out_dim = 16
            fc_out_dim = 33
            cnn_out_dim = 11
            hid_out_dim = 33

            criterion = "L1"
            learning_rate = 1e-3
            epochs = 10000
            print_every = 1000
            average_iter = 100
            device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

        args = Args()
        #### reseat args finished fuck
        
        for epoch in [400]:
            cnn_embedding = np.load(os.path.join(save_folder, f'cnn_{a}_testimgsize:{a}_l1_layer6_byepoch' ,f"cnn_embedding_{a}_full_l1_dim128_epoch{epoch}.npy"))
            cnn_embedding = torch.from_numpy(cnn_embedding).float().to(args.device)

            cnn = cnn_embedding
            for c in cri:
                args.criterion = c
                model = SNAP_GNN(args)
                
                print([features.shape, cnn_embedding.shape, combo_nbhd.shape])
                new_loss = train_gnn_get_testloss(features = features, cnn_feature = cnn_embedding, feat_edge_index = feat_edge_index,
                           spat_edge_index = spat_edge_index, cell_nbhd = combo_nbhd, train_mask = train_mask,
                           model = model, 
                           args = args, verbose = False)
                
                d["imgsize"].append(a)
                d["Loss"].append(new_loss)
                d["Loss_type"].append(c)
                d["Model"].append("cellsnap")

                print(new_loss, a)

df = pd.DataFrame.from_dict(d)
df
df.to_csv("./imgsize_lost_spleen.csv")
                
                