import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import utils
import graph
import preprocessing
import anndata as ad
import scanpy as sc


class CellSNAP:

    def __init__(self,
                 dataset,
                 device,
                 cnn_model=None,
                 cnn_latent_dim=128,
                 gnn_latent_dim=32):
        self.dataset = dataset
        self.device = device
        self.output_dim = self.dataset.cell_nbhd.shape[1]
        self.gnn_latent_dim = gnn_latent_dim
        self.cnn_latent_dim = cnn_latent_dim
        if self.cnn_model:
            self.cnn_model = SNAP_CNN(cnn_latent_dim, self.output_dim)
            self.gnn_model = SNAP_GNN(gnn_latent_dim, self.output_dim)
        else:
            self.gnn_model = SNAP_GNN_simple(gnn_latent_dim,
                                             self.cnn_latent_dim,
                                             self.output_dim)
        return

    def fit_snap_cnn(self,
                     batch_size=64,
                     learning_rate=1e-4,
                     n_epochs=300,
                     loss_fn='MSELoss',
                     OptimizerAlg='Adam',
                     optimizer_kwargs=None,
                     SchedulerAlg=None,
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
        criterion.to(device)
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

    def pred_cnn_embedding(self, batch_size=512, path2result=None):
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

    def fit_snap_gnn(self,
                     learning_rate=1e-3,
                     n_epochs=3000,
                     loss_fn='MSELoss',
                     OptimizerAlg='Adam',
                     optimizer_kwargs=None,
                     SchedulerAlg=None,
                     scheduler_kwargs=None,
                     print_every=500,
                     verbose=True):
        features = torch.from_numpy(self.dataset.features).float().to(
            self.device)
        feature_edges = self.feature_edges
        edge_index = torch.from_numpy(np.array(feature_edges[:2])).long().to(
            self.device)
        cell_nbhd = torch.from_numpy(self.dataset.cell_nbhd).float().to(
            self.device)
        if loss_fn == 'MSELoss':
            criterion = nn.MSELoss()
        elif loss_fn == 'L1Loss':
            criterion = nn.L1Loss()
        elif loss_fn == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        criterion.to(device)
        self.gnn_model.to(device)
        self.gnn_model.train()
        cnn_embedding = torch.from_numpy(self.cnn_embedding).float().to(
            args.device)
        for e in range(1, 1 + n_epochs):
            if self.cnn_model != None:
                predicted_nbhd = self.gnn_model(x=features,
                                                cnn_embed=cnn_embedding,
                                                edge_index=edge_index)
            else:
                predicted_nbhd = self.gnn_model(x=features,
                                                edge_index=edge_index)
            # Compute prediction error
            loss = criterion(predicted_nbhd, train_nbhd)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # take one step
            optimizer.step()
            if scheduler:
                scheduler.step()

            # record the loss
            curr_train_loss = loss.item()
            if verbose and e % print_every == 0:
                print(
                    f'===Epoch {e}, the training loss is {curr_train_loss:>0.8f}==',
                    flush=True)

        print('\nSave CellSNAP Embedding!============\n', flush=True)
        model.eval()
        with torch.no_grad():
            if self.cnn_model != None:
                self.embedding = self.gnn_model(x=features,
                                                cnn_embed=cnn_embedding,
                                                edge_index=edge_index)
            else:
                self.embedding = self.gnn_model(x=features,
                                                edge_index=edge_index)

        return

    def fit_transform(self,
                      cnn_batch_size=64,
                      cnn_learning_rate=1e-4,
                      cnn_epochs=300,
                      cnn_loss_fn='MSELoss',
                      cnn_print=10,
                      gnn_learning_rate=1e-3,
                      gnn_epochs=3000,
                      gnn_loss_fn='MSELoss',
                      optim='Adam',
                      optim_kwargs=None,
                      sche=None,
                      sche_kwargs=None,
                      gnn_print=10,
                      verbose=True):
        if cnn_model:
            self.fit_snap_cnn(self,
                              batch_size=cnn_batch_size,
                              learning_rate=cnn_learning_rate,
                              n_epochs=cnn_epochs,
                              loss_fn=cnn_loss_fn,
                              OptimizerAlg=optim,
                              optimizer_kwargs=optim_kwargs,
                              SchedulerAlg=sche,
                              scheduler_kwargs=sche_kwargs,
                              print_every=cnn_print)
        fit_snap_gnn(self,
                     learning_rate=gnn_learning_rate,
                     n_epochs=gnn_epochs,
                     loss_fn=gnn_loss_fn,
                     OptimizerAlg=optim,
                     optimizer_kwargs=optim_kwargs,
                     SchedulerAlg=sche,
                     scheduler_kwargs=sche_kwargs,
                     print_every=cnn_print,
                     verbose=verbose)

        return

    def visualize_umap(self, embedding, label):
        # visualization of umap of the embedding
        adata = ad.AnnData(embedding)
        feature_adata.obs['annotation'] = list(label)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10)
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (10, 8)
        sc.pl.umap(adata, color='annotation', legend_fontsize=17, show=False)
        plt.show()
        return
