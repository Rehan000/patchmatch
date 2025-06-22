import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMatchEncoder(nn.Module):
    """
    Lightweight encoder that transforms grayscale image patches into embedding vectors
    using depthwise separable convolutions, batch normalization, and global average pooling.

    Designed to be used as the shared encoder in a Triplet network.
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
            nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),  # (B, 32, 40, 40)
            nn.ReLU(),
            nn.BatchNorm2d(48),

            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2),  # (B, 64, 20, 20)

            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=1),  # Pointwise
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
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class PatchMatchTripletNetwork(nn.Module):
    """
    Triplet network built on a shared PatchMatchEncoder for learning embeddings
    that cluster similar patches and push apart dissimilar ones using Triplet Loss.
    """

    def __init__(self, input_channels=1, embedding_dim=64):
        """
        Initializes the triplet network.

        Args:
            input_channels (int): Number of input channels.
            embedding_dim (int): Size of the output descriptor vector.
        """
        super(PatchMatchTripletNetwork, self).__init__()
        self.encoder = PatchMatchEncoder(input_channels, embedding_dim)

    def forward(self, anchor, positive, negative):
        """
        Computes embeddings for anchor, positive, and negative patches.

        Args:
            anchor (Tensor): Anchor patch, shape (B, 1, 40, 40)
            positive (Tensor): Positive patch, shape (B, 1, 40, 40)
            negative (Tensor): Negative patch, shape (B, 1, 40, 40)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized embeddings for anchor, positive, and negative.
        """
        anchor_embed = self.encoder(anchor)
        positive_embed = self.encoder(positive)
        negative_embed = self.encoder(negative)
        return anchor_embed, positive_embed, negative_embed


class TripletLoss(nn.Module):
    """
    Triplet margin loss for embedding learning.

    Encourages anchor-positive distances to be smaller than anchor-negative distances
    by a specified margin.
    """

    def __init__(self, margin=1.0):
        """
        Initializes the triplet loss module.

        Args:
            margin (float): Margin by which negatives should be farther than positives.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor_embed, positive_embed, negative_embed):
        """
        Computes the triplet loss.

        Args:
            anchor_embed (Tensor): Anchor embeddings, shape (B, D)
            positive_embed (Tensor): Positive embeddings, shape (B, D)
            negative_embed (Tensor): Negative embeddings, shape (B, D)

        Returns:
            Tensor: Scalar loss value.
        """
        return self.loss_fn(anchor_embed, positive_embed, negative_embed)
