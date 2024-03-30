import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SNAP_GNN_LITE(nn.Module):

    def __init__(self, input_dim, out_dim, gnn_latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=gnn_latent_dim)
        self.gnn_conv1 = GCNConv(gnn_latent_dim, gnn_latent_dim)
        self.gnn_conv2 = GCNConv(gnn_latent_dim, out_dim * 2)

    def encoder(self, x, edge_index):
        x = F.relu(self.fc(x))
        x = self.gnn_conv1(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = F.relu(self.encoder(x, edge_index))
        x = self.gnn_conv2(x, edge_index)
        return x


class SNAP_GNN(nn.Module):

    def __init__(self,
                 feature_input_dim=32,
                 cnn_latent_dim=128,
                 gnn_latent_dim=32,
                 out_dim=20):
        super().__init__()
        self.fc = nn.Linear(in_features=feature_input_dim,
                            out_features=gnn_latent_dim)
        self.cnn_fc = nn.Linear(in_features=cnn_latent_dim,
                                out_features=gnn_latent_dim)
        self.gnn_conv1 = GCNConv(2 * gnn_latent_dim, gnn_latent_dim)
        self.gnn_conv2 = GCNConv(gnn_latent_dim, out_dim)

    def encoder(self, x, cnn_embed, edge_index):
        cnn_feat = self.cnn_fc(cnn_embed)
        x = F.relu(torch.cat((self.fc(x), cnn_feat), dim=1))
        x = self.gnn_conv1(x, edge_index)
        return x

    def forward(self, x, cnn_embed, edge_index):
        x = F.relu(self.gnn_encoder(x, cnn_embed, edge_index))
        x = self.gnn_conv2(x, edge_index)
        return x


class SNAP_GNN_DUO(nn.Module):

    def __init__(self,
                 out_dim,
                 feature_input_dim,
                 cnn_input_dim,
                 gnn_latent_dim=33,
                 proj_dim=32,
                 fc_out_dim=33,
                 cnn_out_dim=11):
        super().__init__()
        self.fc = nn.Linear(in_features=feature_input_dim,
                            out_features=proj_dim)
        self.cnn_fc = nn.Linear(in_features=cnn_input_dim,
                                out_features=proj_dim)
        self.feat_conv1 = GCNConv(proj_dim, proj_dim)
        self.feat_conv2 = GCNConv(proj_dim, fc_out_dim)

        self.spat_conv1 = GCNConv(proj_dim, proj_dim)
        self.spat_conv2 = GCNConv(proj_dim, cnn_out_dim)

        self.proj1 = nn.Linear(in_features=fc_out_dim + cnn_out_dim,
                               out_features=gnn_latent_dim)
        self.proj2 = nn.Linear(in_features=gnn_latent_dim,
                               out_features=out_dim * 2)

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
        x = torch.cat((x_feat, x_spat), dim=1)
        return x

    def forward(self, feat, spat, feat_edge_index, spat_edge_index):
        x = F.relu(self.encoder(feat, spat, feat_edge_index, spat_edge_index))
        x = self.proj1(x)
        x = F.relu(x)
        x = self.proj2(x)
        return x


class SNAP_CNN(nn.Module):

    def __init__(self, cnn_latent_dim, output_dim):
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
        self.fc3 = nn.Linear(512, cnn_latent_dim)
        self.fc4 = nn.Linear(cnn_latent_dim, output_dim)

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
