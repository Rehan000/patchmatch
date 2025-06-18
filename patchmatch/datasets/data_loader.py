import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PatchTripletDataset(Dataset):
    """
    Loads triplet patch datasets from .npz files and provides samples for training
    using Triplet Loss in PyTorch.

    Each sample is a tuple: (anchor, positive, negative)
    """

    def __init__(self, npz_path, transform=None):
        """
        Initializes the dataset.

        Args:
            npz_path (str): Path to the .npz file containing 'triplets'.
            transform (callable, optional): Optional transform to be applied to each patch.
        """
        self.npz_path = npz_path
        self.transform = transform
        self._load_data()

    def _load_data(self):
        """
        Loads and preprocesses triplet patches from the .npz file.
        """
        data = np.load(self.npz_path, mmap_mode='r')
        triplets = data['triplets']  # shape: (N, H, W, 3)

        triplets = triplets.astype(np.float32) / 255.0
        anchors     = triplets[..., 0]  # (N, H, W)
        positives   = triplets[..., 1]  # (N, H, W)
        negatives   = triplets[..., 2]  # (N, H, W)

        self.anchors   = torch.from_numpy(anchors).unsqueeze(1)   # (N, 1, H, W)
        self.positives = torch.from_numpy(positives).unsqueeze(1) # (N, 1, H, W)
        self.negatives = torch.from_numpy(negatives).unsqueeze(1) # (N, 1, H, W)

    def __len__(self):
        """
        Returns the total number of triplets.
        """
        return self.anchors.shape[0]

    def __getitem__(self, idx):
        """
        Returns the idx-th triplet as (anchor, positive, negative).

        Returns:
            tuple: (anchor_tensor, positive_tensor, negative_tensor)
        """
        a = self.anchors[idx]
        p = self.positives[idx]
        n = self.negatives[idx]

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n
