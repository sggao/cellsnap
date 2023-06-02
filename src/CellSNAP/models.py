import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SNAP_GNN_simple(nn.Module):
    def __init__(self, input_dim, fc_dim, latent_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=fc_dim)
        self.gnn_conv1 = GCNConv(fc_dim, latent_dim)
        self.gnn_conv2 = GCNConv(latent_dim, out_dim)

    def gnn_encoder(self, x, edge_index):
        x = F.relu(self.fc(x))
        x = self.gnn_conv1(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = F.relu(self.gnn_encoder(x, edge_index))
        x = self.gnn_conv2(x, edge_index)
        return x


class SNAP_GNN(nn.Module):
    def __init__(
            self, feature_input_dim=32, feature_latent_dim=24,
            img_input_dim=32, img_latent_dim=8, gnn_latent_dim=32, out_dim=20
    ):
        super().__init__()
        self.img_encoder = nn.Linear(in_features=img_input_dim, out_features=img_latent_dim)
        self.feature_encoder = nn.Linear(in_features=feature_input_dim, out_features=feature_latent_dim)
        self.gnn_conv1 = GCNConv(feature_latent_dim+img_latent_dim, gnn_latent_dim)
        self.gnn_conv2 = GCNConv(gnn_latent_dim, out_dim)

    def gnn_encoder(self, feature, img, edge_index):
        feature = F.relu(self.feature_encoder(feature))
        img = F.relu(self.img_encoder(img))
        x = torch.cat((feature, img), 1)
        embedding = self.gnn_conv1(x, edge_index)
        return embedding

    def forward(self, feature, img, edge_index):
        output = F.relu(self.gnn_encoder(feature, img, edge_index))
        output = self.gnn_conv2(output, edge_index)
        return output