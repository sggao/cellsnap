import torch
from torch.utils.data import Dataset
import sys
sys.path.append("../../../../../")

from torchvision import transforms
from preprocess_codex_tonsil_data import *
import graph

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
from resnet_models import *

## test knn 20

load_path = '/mnt/cloud1/sheng-projects/st_projects/spatial_clust/spatial-clust-scripts/ipynb/Bokai_reorg/benchmark/spleen/data/'
#df_clean = pd.read_csv(os.path.join(load_path, ""), index_col=0)
cell_nbhd = np.load(os.path.join(load_path, "cell_nbhd_res0.5_k20.npy"))
train_mask = np.load(os.path.join(load_path, "train_mask.npy"))
cell_nbhd.shape

class SingleCellImageDataset_Stream(Dataset):
    def __init__(self, img_path, train_mask, cell_nbhd, use_transform):
        """
        Form dataset of single cells
        Parameters
        ----------
        images: np.ndarray of shape (n_samples, C, H, W)
        cell_nbhd: np.ndarray of shape (n_samples, d)
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        self.labels = cell_nbhd
        self.use_transform = use_transform
        self.img_path = img_path
        self.train_list = np.arange(train_mask.shape[0])[train_mask]
        self.test_list = np.arange(train_mask.shape[0])[~train_mask]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.use_transform:
            ind = self.train_list[index]
        else:
            ind = self.test_list[index]
        img = np.load(os.path.join(self.img_path, f"img_{ind:05d}.npy"))
        if self.use_transform:
            img = self.transform(torch.Tensor(img))
        labels = self.labels[index]
        return img, labels
    
load_path2 = '/mnt/cloud1/sheng-projects/st_projects/spatial_clust/data/codex_murine/'

alpha = 0.9 # default
size = 512 # default

train_nbhd = cell_nbhd[train_mask, :]
test_nbhd = cell_nbhd[~train_mask, :]

image_folder = os.path.join(load_path2, "processed_data", "single_cell_images", f"size{size}_qr{alpha}")
train_dataset = SingleCellImageDataset_Stream(os.path.join(image_folder, "images"), train_mask, train_nbhd, use_transform = True)
test_dataset = SingleCellImageDataset_Stream(os.path.join(image_folder, "images"), train_mask, test_nbhd, use_transform = False)

train_size, test_size = train_nbhd.shape[0], test_nbhd.shape[0]

class CNN_cell512_channel2_layer6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size = 2, stride = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 2, stride = 1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 2, stride = 1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size = 2, stride = 1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, cell_nbhd.shape[1])
    
    def cnn_encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.fc4(F.relu(x))
        return x
    
epochs = 500
batch_size = 64
learning_rate = 1e-4
model = "cnn"
if model == "cnn":
    cnn_net = CNN_cell512_channel2_layer6()
else:
    cnn_net = resnet12()
    
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

l = 1
if l == 1:
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_net.parameters(), lr=learning_rate)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cnn_net = cnn_net.to(device)
criterion = criterion.to(device)

train_size, test_size = train_nbhd.shape[0], test_nbhd.shape[0]
model_save_path = os.path.join(load_path, "cnn", f"cnn_512_l{l}_layer6_testnbsize:20_checkpoints", "epochs")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    
train_loss_epoch = []
test_loss_epoch = []

for epoch in range(1, 1+epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_sample = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device).to(torch.float32)
        labels = labels.to(device).to(torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item() * inputs.shape[0]
        running_sample += inputs.shape[0]
        if i %100 == 99:    # print every 2000 mini-batches
            print(f'===Epoch {epoch}, Step {i + 1:5d} loss: {running_loss / running_sample:.6f}===')
            #running_loss = 0.0
    
    test_loss = 0.0
    test_sample = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device).to(torch.float32)
            outputs = cnn_net(inputs)
            labels = labels.to(device).to(torch.float32)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.shape[0]
            test_sample += inputs.shape[0]
    assert(test_sample == test_size)
    assert(running_sample == train_size)
    print(f'{epoch} Training Loss: {running_loss/train_size}')
    print(f'{epoch} Validation Loss: {test_loss/test_size}')
    
    train_loss_epoch.append(running_loss/train_size)
    test_loss_epoch.append(test_loss/test_size)
    if epoch % 50 == 0:
        torch.save(cnn_net.state_dict(), os.path.join(model_save_path, f"epoch{epoch}_model_weights.pth"))

            

print('Finished Training')    

