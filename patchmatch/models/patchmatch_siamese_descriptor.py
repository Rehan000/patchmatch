import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMatchEncoder(nn.Module):
    """
    Lightweight encoder that transforms grayscale image patches into embedding vectors
    using depthwise separable convolutions, batch normalization, and global average pooling.

    Designed to be used as the shared encoder in a Siamese network.
    """

    def __init__(self, input_channels=1, embedding_dim=64):
        """
        Initializes the encoder module.

        Args:
            input_channels (int): Number of input channels (default 1 for grayscale).
            embedding_dim (int): Size of the output embedding vector.
        """
        super(PatchMatchEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # (B, 32, 40, 40)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # (B, 64, 20, 20)

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.AdaptiveAvgPool2d(1),  # (B, 64, 1, 1)
        )

        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, 1, 40, 40).

        Returns:
            Tensor: L2-normalized embedding of shape (B, embedding_dim).
        """
        x = self.encoder(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.projection(x)  # (B, embedding_dim)
        x = F.normalize(x, p=2, dim=1)
        return x


class PatchMatchSiameseDescriptor(nn.Module):
    """
    Siamese network built on a shared PatchMatchEncoder for comparing image patches.
    The network computes the L2 distance between descriptors of two input patches.
    """

    def __init__(self, input_channels=1, embedding_dim=64):
        """
        Initializes the Siamese network.

        Args:
            input_channels (int): Number of input channels (default 1 for grayscale).
            embedding_dim (int): Size of the output descriptor vector.
        """
        super(PatchMatchSiameseDescriptor, self).__init__()
        self.encoder = PatchMatchEncoder(input_channels, embedding_dim)

    def forward(self, x1, x2):
        """
        Forward pass to compute the Euclidean distance between embeddings.

        Args:
            x1 (Tensor): First patch batch, shape (B, 1, 40, 40).
            x2 (Tensor): Second patch batch, shape (B, 1, 40, 40).

        Returns:
            Tensor: Euclidean distance between embeddings, shape (B, 1).
        """
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        distance = torch.norm(feat1 - feat2, dim=1, keepdim=True)
        return distance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training Siamese networks.

    Encourages the network to produce embeddings with small distances for
    similar pairs and larger distances (greater than a margin) for dissimilar pairs.
    """

    def __init__(self, margin=1.0):
        """
        Initializes the contrastive loss module.

        Args:
            margin (float): Distance margin for dissimilar pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances, labels):
        """
        Computes the contrastive loss.

        Args:
            distances (Tensor): Euclidean distances between patch embeddings, shape (B, 1).
            labels (Tensor): Ground truth labels (1 for similar, 0 for dissimilar), shape (B,).

        Returns:
            Tensor: Scalar loss value.
        """
        labels = labels.float()
        loss = labels * distances.pow(2) + (1 - labels) * F.relu(self.margin - distances).pow(2)
        return loss.mean()
