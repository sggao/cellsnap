import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SingleCellImageDataset(Dataset):
    def __init__(self, images):
        """
        Form dataset of single cells
        Parameters
        ----------
        images: np.ndarray of shape (n_samples, C, H, W)
        """
        super().__init__()
        self.images = torch.from_numpy(images).float()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = self.images[index, :, :, :]
        img = self.transform(img)
        return img



