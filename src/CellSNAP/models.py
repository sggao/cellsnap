import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class simple_SNAP_GNN(nn.Module):

    def __init__(self, feature_input_dim=32, gnn_latent_dim=32, out_dim=20):
        super().__init__()
        self.fc = nn.Linear(in_features=feature_input_dim,
                            out_features=gnn_latent_dim)
        self.gnn_conv1 = GCNConv(gnn_latent_dim, gnn_latent_dim)
        self.gnn_conv2 = GCNConv(gnn_latent_dim, out_dim)

    def gnn_encoder(self, x, edge_index):
        x = F.relu(self.fc(x))
        x = self.gnn_conv1(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = F.relu(self.gnn_encoder(x, edge_index))
        x = self.gnn_conv2(x, edge_index)
        return x


class SNAP_GNN(nn.Module):

    def __init__(self, feature_input_dim=32, cnn_latent_dim = 128, gnn_latent_dim=32, out_dim=20):
        super().__init__()
        self.fc = nn.Linear(in_features=feature_input_dim,
                            out_features=gnn_latent_dim)
        self.cnn_fc = nn.Linear(in_features=cnn_latent_dim,
                                out_features=gnn_latent_dim)
        self.gnn_conv1 = GCNConv(2 * gnn_latent_dim, gnn_latent_dim)
        self.gnn_conv2 = GCNConv(gnn_latent_dim, out_dim)

    def gnn_encoder(self, x, cnn_embed, edge_index):
        cnn_feat = self.cnn_fc(cnn_embed)
        x = F.relu(torch.cat((self.fc(x), cnn_feat), dim=1))
        x = self.gnn_conv1(x, edge_index)
        return x

    def forward(self, x, cnn_embed, edge_index):
        x = F.relu(self.gnn_encoder(x, cnn_embed, edge_index))
        x = self.gnn_conv2(x, edge_index)
        return x


class SNAP_CNN(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=2, stride=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def cnn_encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.fc4(F.relu(x))
        return x
