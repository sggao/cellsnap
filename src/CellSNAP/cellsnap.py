import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import utils
import graph
import preprocessing


class CellSNAP:

    def __init__(self,
                 dataset,
                 device,
                 gnn_model,
                 cnn_model=None,
                 cnn_latent_dim=128,
                 gnn_latent_dim=32):
        self.dataset = dataset
        self.device = device
        self.output_dim = self.dataset.cell_nbhd.shape[1]
        self.gnn_latent_dim = gnn_latent_dim
        self.cnn_latent_dim = cnn_latent_dim
        self.gnn_model = gnn_model(gnn_latent_dim, self.output_dim)
        if self.cnn_model:
            self.cnn_model = cnn_model(cnn_latent_dim, self.output_dim)
        return

    def train_snap_cnn(self,
                       batch_size=64,
                       learning_rate=1e-4,
                       n_epochs=300,
                       loss_fn='MSELoss',
                       OptimizerAlg='Adam',
                       optimizer_kwargs=None,
                       SchedulerAlg='StepLR',
                       scheduler_kwargs=None,
                       print_every=10):
        print('\nTraining convolutional neural network============\n',
              flush=True)
        # enable data augmentation
        self.dataset.use_transform = True
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=1)
        if loss_fn == 'MSELoss':
            criterion = nn.MSELoss()
        elif loss_fn == 'L1Loss':
            criterion = nn.L1Loss()
        elif loss_fn == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        self.cnn_model.to(self.device)
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.cnn_model.parameters(), OptimizerAlg, optimizer_kwargs,
            SchedulerAlg, scheduler_kwargs)
        self.cnn_model.train()
        for epoch in range(1, 1 +
                           n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_sample = 0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device).to(torch.float32)
                labels = labels.to(self.device).to(torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.cnn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # change learning rate
                if scheduler:
                    scheduler.step()

                # print statistics
                running_loss += loss.item() * inputs.shape[0]
                running_sample += inputs.shape[0]
            if epoch % print_every == 0:  # print every 2000 mini-batches
                print(
                    f'===Epoch {epoch}, Step {i + 1:5d} loss: {running_loss / running_sample:.6f}==='
                )

        return

    def calculate_cnn_embedding(self, batch_size=512, path2result=None):
        self.dataset.use_transform = False
        testloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1)
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        self.cnn_embedding = np.zeros(
            (self.dataset.df.shape[0], self.cnn_latent_dim))
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device).to(torch.float32)
                outputs = self.cnn_model.cnn_encoder(inputs)
                self.cnn_embedding[start_idx:start_idx +
                                   inputs.shape[0]] = outputs.cpu()

                start_idx += inputs.shape[0]
        assert (start_idx == self.dataset.df.shape[0])
        print('\nSave CNN Embedding!============\n', flush=True)
        if path2result:
            if not os.path.exists(path2result):
                os.makedirs(path2result)
            np.save(os.path.join(path2result, "SNAP_CNN_embedding.npy"),
                    self.cnn_embedding)

    def train_snap_gnn(self, dataset):
        
        return

    def fit_transform(self, dataset):
        return

    def visualize_umap(self, dataset):
        return
