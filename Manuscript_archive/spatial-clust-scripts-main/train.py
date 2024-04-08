import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

# import data
import model


def fit_transform(
        features, edge_index, cell_nbhd,
        feature_latent_dim, gnn_latent_dim,
        images=None, img_latent_dim=None,
        n_epochs=300, loss_fn='MSELoss',
        OptimizerAlg='Adam', optimizer_kwargs=None,
        SchedulerAlg='StepLR', scheduler_kwargs=None,
        device='cpu',
        print_every=10
):
    if images is None:
        # only train GNN
        print('Training graph convolutional network============\n', flush=True)
        gcn = model.GCN(
            input_dim=features.shape[1],
            fc_dim=feature_latent_dim,
            latent_dim=gnn_latent_dim,
            out_dim=cell_nbhd.shape[1]
        )
        train_gcn(
            model=gcn,
            features=features, edge_index=edge_index, cell_nbhd=cell_nbhd,
            n_epochs=n_epochs, loss_fn=loss_fn,
            OptimizerAlg=OptimizerAlg, optimizer_kwargs=optimizer_kwargs,
            SchedulerAlg=SchedulerAlg, scheduler_kwargs=scheduler_kwargs,
            device=device, print_every=print_every
        )
        features = torch.from_numpy(features).float().to(device)
        edge_index = torch.from_numpy(edge_index).long().to(device)

        print('\nExtracting embeddings============\n', flush=True)
        gcn.eval()
        with torch.no_grad():
            gnn_embeddings = gcn.gnn_encoder(features, edge_index)
        gnn_embeddings = gnn_embeddings.to('cpu').detach().numpy()

        print('\nDone!', flush=True)
        return gnn_embeddings, gcn

    # train CNN+GNN
    conv_gcn = model.ConvGCN(
        feature_input_dim=features.shape[1],
        feature_latent_dim=feature_latent_dim,
        img_input_dim=images.shape[1],
        img_latent_dim=img_latent_dim,
        gnn_latent_dim=gnn_latent_dim,
        out_dim=cell_nbhd.shape[1]
    )

    print('\nTraining convolutional graph convolutional neural network============\n', flush=True)
    train_conv_gcn(
        model=conv_gcn,
        features=features,
        images=images,
        edge_index=edge_index,
        cell_nbhd=cell_nbhd,
        n_epochs=n_epochs,
        loss_fn=loss_fn,
        OptimizerAlg=OptimizerAlg, optimizer_kwargs=optimizer_kwargs,
        SchedulerAlg=SchedulerAlg, scheduler_kwargs=scheduler_kwargs,
        device=device,
        print_every=print_every
    )

    print('\nExtracting embeddings============\n', flush=True)

    features = torch.from_numpy(features).float().to(device)
    images = torch.from_numpy(images).float().to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    conv_gcn.eval()
    with torch.no_grad():
        gnn_embeddings = conv_gcn.gnn_encoder(
            feature=features,
            img=images,
            edge_index=edge_index
        )
        img_embeddings = conv_gcn.img_encoder(images)

    gnn_embeddings = gnn_embeddings.to('cpu').detach().numpy()
    img_embeddings = img_embeddings.to('cpu').detach().numpy()

    print('\nDone!', flush=True)

    return gnn_embeddings, img_embeddings, conv_gcn


def get_optimizer_and_scheduler(
        parameters,
        OptimizerAlg='Adam', optimizer_kwargs=None,
        SchedulerAlg='StepLR', scheduler_kwargs=None
):
    if SchedulerAlg == "StepLR":
        SchedulerAlg = optim.lr_scheduler.StepLR
    elif SchedulerAlg == "MultiStepLR":
        SchedulerAlg = optim.lr_scheduler.MultiStepLR
    else:
        raise NotImplementedError

    if OptimizerAlg == "SGD":
        OptimizerAlg = optim.SGD
    elif OptimizerAlg == "Adadelta":
        OptimizerAlg = optim.Adadelta
    elif OptimizerAlg == "Adam":
        OptimizerAlg = optim.Adam
    elif OptimizerAlg == "RMSprop":
        OptimizerAlg = optim.RMSprop
    else:
        raise NotImplementedError

    if optimizer_kwargs is not None:
        optimizer = OptimizerAlg(parameters, **optimizer_kwargs)
    else:
        optimizer = OptimizerAlg(parameters)

    if scheduler_kwargs is not None:
        scheduler = SchedulerAlg(optimizer, **scheduler_kwargs)
    else:
        scheduler = SchedulerAlg(optimizer)

    return optimizer, scheduler


def train_gcn(
        model, features, edge_index, cell_nbhd,
        n_epochs=100, loss_fn='CrossEntropyLoss',
        OptimizerAlg='Adam', optimizer_kwargs=None,
        SchedulerAlg='StepLR', scheduler_kwargs=None,
        device='cpu', print_every=10
):
    print('Start training ===========')
    if loss_fn == 'MSELoss':
        loss_fn = nn.MSELoss()
        # additional_transform = F.relu
        additional_transform = lambda x: F.softmax(x, dim=1)
    elif loss_fn == 'KLDivLoss':
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        additional_transform = lambda x: F.log_softmax(x, dim=1)
    elif loss_fn == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        additional_transform = lambda x: x
    else:
        raise NotImplementedError

    model.to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model.parameters(),
        OptimizerAlg, optimizer_kwargs,
        SchedulerAlg, scheduler_kwargs
    )
    features = torch.from_numpy(features).float().to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    cell_nbhd = torch.from_numpy(cell_nbhd).float().to(device)

    model.train()
    for e in range(n_epochs):
        predicted_nbhd = additional_transform(model(features, edge_index))
        # Compute prediction error
        loss = loss_fn(predicted_nbhd, cell_nbhd)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # take one step
        optimizer.step()

        # record the loss
        curr_train_loss = loss.item()
        if e % print_every == 0:
            print(f'After epoch {e}, the training loss is {curr_train_loss:>0.8f}', flush=True)

        # change learning rate
        scheduler.step()


def train_conv_gcn(
        model, features, images, edge_index, cell_nbhd,
        n_epochs=100, loss_fn='CrossEntropyLoss',
        OptimizerAlg='Adam', optimizer_kwargs=None,
        SchedulerAlg='StepLR', scheduler_kwargs=None,
        device='cpu', print_every=10
):
    print('Start training ===========')
    if loss_fn == 'MSELoss':
        loss_fn = nn.MSELoss()
        # additional_transform = F.relu
        additional_transform = lambda x: F.softmax(x, dim=1)
    elif loss_fn == 'KLDivLoss':
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        additional_transform = lambda x: F.log_softmax(x, dim=1)
    elif loss_fn == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        additional_transform = lambda x: x
    else:
        raise NotImplementedError

    model.to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model.parameters(),
        OptimizerAlg, optimizer_kwargs,
        SchedulerAlg, scheduler_kwargs
    )

    images = torch.from_numpy(images).float().to(device)
    features = torch.from_numpy(features).float().to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    cell_nbhd = torch.from_numpy(cell_nbhd).float().to(device)

    model.train()
    for e in range(n_epochs):
        predicted_nbhd = additional_transform(model(feature=features, img=images, edge_index=edge_index))
        # Compute prediction error
        loss = loss_fn(predicted_nbhd, cell_nbhd)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # take one step
        optimizer.step()

        # record the loss
        curr_train_loss = loss.item()
        if e % print_every == 0:
            print(f'After epoch {e}, the training loss is {curr_train_loss:>0.8f}', flush=True)

        # change learning rate
        scheduler.step()


# def add_noise(inputs, noise_factor=0.3):
#     noisy = inputs + torch.randn_like(inputs) * noise_factor
#     noisy = torch.clip(noisy, 0., 1.)
#     return noisy
#
#
# def train_conv_autoencoder_for_one_epoch(
#         dataloader, encoder, decoder, loss_fn, optimizer, noise_factor=0.3, device='cpu'
# ):
#     encoder.train()
#     decoder.train()
#     train_loss = 0.
#     for batch, img in enumerate(dataloader):
#         noisy_img = add_noise(img, noise_factor)
#         img = img.to(device)
#         noisy_img = noisy_img.to(device)
#
#         # Compute prediction error
#         encoded_img = encoder(noisy_img)
#         decoded_img = decoder(encoded_img)
#         loss = loss_fn(decoded_img, img)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#
#         # record the loss and accuracy
#         train_loss += loss.item()
#
#         # take one step
#         optimizer.step()
#
#     train_loss /= len(dataloader)
#
#     return train_loss
#
#
# def train_conv_autoencoder(
#         encoder, decoder, dataloader, n_epochs,
#         loss_fn='MSELoss', noise_factor=0.2,
#         OptimizerAlg='Adam', optimizer_kwargs=None,
#         SchedulerAlg='StepLR', scheduler_kwargs=None,
#         device='cpu', print_every=10
# ):
#     print('Start training ===========')
#     if loss_fn == 'MSELoss':
#         loss_fn = nn.MSELoss()
#     else:
#         raise NotImplementedError
#
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)
#     params_to_optimize = [
#         {'params': encoder.parameters()},
#         {'params': decoder.parameters()}
#     ]
#
#     optimizer, scheduler = get_optimizer_and_scheduler(
#         params_to_optimize,
#         OptimizerAlg, optimizer_kwargs,
#         SchedulerAlg, scheduler_kwargs
#     )
#
#     for e in range(n_epochs):
#         curr_train_loss = train_conv_autoencoder_for_one_epoch(
#             dataloader=dataloader,
#             encoder=encoder, decoder=decoder,
#             loss_fn=loss_fn, optimizer=optimizer, noise_factor=noise_factor, device=device)
#         if e % print_every == 0:
#             print(f'After epoch {e}, the training loss is {curr_train_loss:>0.8f}', flush=True)
#         scheduler.step()
