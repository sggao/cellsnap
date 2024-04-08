import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
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


class ConvGCN(nn.Module):
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


# class MurineConvEncoder(nn.Module):
#     def __init__(self, latent_dim=64):
#         super().__init__()
#         self.cnn_encoder = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2),  # 8*15*15
#             nn.BatchNorm2d(num_features=8),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),  # 16*7*7
#             nn.BatchNorm2d(num_features=16),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),  # 32*3*3
#             nn.ReLU(True)
#         )
#         self.flatten = nn.Flatten(start_dim=1)
#         self.lin_encoder = nn.Sequential(
#             nn.Linear(in_features=32*3*3, out_features=128),
#             nn.ReLU(True),
#             nn.Linear(in_features=128, out_features=latent_dim),
#         )
#
#     def forward(self, x):
#         x = self.cnn_encoder(x)
#         x = self.flatten(x)
#         x = self.lin_encoder(x)
#         return x
#
#
# class TonsilConvEncoder(nn.Module):
#     def __init__(self, latent_dim=8):
#         super().__init__()
#         self.cnn_encoder = nn.Sequential(
#             # 2*55*55
#             nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2),  # 8*27*27
#             nn.BatchNorm2d(num_features=8),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # 8*13*13
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),  # 16*6*6
#             nn.BatchNorm2d(num_features=16),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),  # 32*2*2
#             nn.ReLU(True)
#         )
#         self.flatten = nn.Flatten(start_dim=1)
#         self.lin_encoder = nn.Sequential(
#             nn.Linear(in_features=32*2*2, out_features=32),
#             nn.ReLU(True),
#             nn.Linear(in_features=32, out_features=latent_dim),
#         )
#
#     def forward(self, x):
#         x = self.cnn_encoder(x)
#         x = self.flatten(x)
#         x = self.lin_encoder(x)
#         return x

#
# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim=64):
#         super().__init__()
#         self.lin_decoder = nn.Sequential(
#             nn.Linear(in_features=latent_dim, out_features=128),
#             nn.ReLU(True),
#             nn.Linear(in_features=128, out_features=32*3*3),
#             nn.ReLU(True)
#         )
#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
#         self.cnn_decoder = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),  # 16*7*7
#             nn.BatchNorm2d(num_features=16),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),  # 8*15*15
#             nn.BatchNorm2d(num_features=8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2, output_padding=1),  # 2*32*32
#         )
#
#     def forward(self, x):
#         x = self.lin_decoder(x)
#         x = self.unflatten(x)
#         x = self.cnn_decoder(x)
#         x = torch.sigmoid(x)
#         return x
