import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PatchTripletDataset(Dataset):
    """
    Lazily loads triplet patch datasets from .npz files for memory-efficient training.

    Each sample is a triplet of patches:
        - anchor: original patch from reference image
        - positive: corresponding patch from warped image
        - negative: a spatially offset patch (either hard or random negative)

    The .npz file is expected to contain a single key 'triplets' with shape (N, H, W, 3),
    where each triplet has 3 grayscale patches stacked along the last axis.
    """

    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path (str): Path to the .npz file containing the triplet data.
            transform (callable, optional): A transform function to apply to each patch.
        """
        self.npz_path = npz_path
        self.transform = transform

        # Use memory-mapped file loading to reduce RAM usage
        self.data = np.load(self.npz_path, mmap_mode='r')
        if 'triplets' not in self.data:
            raise KeyError(f"[ERROR] 'triplets' key not found in: {npz_path}")

        self.triplets = self.data['triplets']  # Shape: (N, H, W, 3)
        self.length = self.triplets.shape[0]

    def __len__(self):
        """
        Returns:
            int: Total number of triplet samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Returns the idx-th triplet as tensors scaled to [0, 1].

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (anchor, positive, negative) each as a torch.FloatTensor of shape (1, H, W)
        """
        triplet = self.triplets[idx]  # (H, W, 3)

        # Split channels and normalize
        anchor = torch.tensor(triplet[..., 0], dtype=torch.float32).unsqueeze(0) / 255.0
        positive = torch.tensor(triplet[..., 1], dtype=torch.float32).unsqueeze(0) / 255.0
        negative = torch.tensor(triplet[..., 2], dtype=torch.float32).unsqueeze(0) / 255.0

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


def create_dataloader(npz_path, batch_size, shuffle=True, num_workers=2):
    """
    Creates a PyTorch DataLoader for the given dataset.

    Args:
        npz_path (str): Path to the .npz file with triplet data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset at the start of each epoch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader yielding batches of (anchor, positive, negative).
    """
    dataset = PatchTripletDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
