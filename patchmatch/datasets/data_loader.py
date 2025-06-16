import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PatchPairDataset(Dataset):
    """
    Loads patch pair datasets from .npz files and provides samples for training
    Siamese networks in PyTorch.

    Each sample is a tuple: ((patch1, patch2), label)
    """

    def __init__(self, npz_path, transform=None):
        """
        Initializes the dataset.

        Args:
            npz_path (str): Path to the .npz file containing 'patches' and 'labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.npz_path = npz_path
        self.transform = transform
        self._load_data()

    def _load_data(self):
        """
        Loads the patches and labels from the .npz file and preprocesses them.
        """
        data = np.load(self.npz_path, mmap_mode='r')
        patches = data['patches']  # shape: (N, H, W, 2)
        labels = data['labels']    # shape: (N,)

        patches = patches.astype(np.float32) / 255.0
        patch1 = patches[..., 0]  # (N, H, W)
        patch2 = patches[..., 1]  # (N, H, W)

        self.patch1 = torch.from_numpy(patch1).unsqueeze(1)  # (N, 1, H, W)
        self.patch2 = torch.from_numpy(patch2).unsqueeze(1)  # (N, 1, H, W)

        labels = labels.astype(np.float32)
        labels[labels == -1] = 0  # Convert hard negatives to negatives
        self.labels = torch.from_numpy(labels)               # (N,)

    def __len__(self):
        """
        Returns the total number of patch pairs.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the idx-th patch pair and label.

        Returns:
            tuple: ((patch1, patch2), label)
        """
        p1 = self.patch1[idx]
        p2 = self.patch2[idx]
        label = self.labels[idx]

        if self.transform:
            p1 = self.transform(p1)
            p2 = self.transform(p2)

        return (p1, p2), label
