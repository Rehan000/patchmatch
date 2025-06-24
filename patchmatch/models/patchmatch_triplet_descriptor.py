import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PatchMatchEncoder(nn.Module):
    """
    Optimized encoder with SE blocks and group convolutions for improved edge performance.
    """
    def __init__(self, input_channels=1, embedding_dim=128):
        super(PatchMatchEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # (B, 64, 40, 40)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1, groups=2),  # Pointwise with groups
            nn.ReLU(),
            nn.BatchNorm2d(128),
            SEBlock(128),  # SE block for attention
            nn.MaxPool2d(kernel_size=2),  # (B, 128, 20, 20)

            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=1, groups=2),  # Pointwise with groups
            nn.ReLU(),
            nn.BatchNorm2d(192),
            SEBlock(192),  # SE block for attention

            # Block 4
            nn.Conv2d(192, 192, kernel_size=3, padding=1, groups=192),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, groups=2),  # Pointwise with groups
            nn.ReLU(),
            nn.BatchNorm2d(192),

            nn.AdaptiveAvgPool2d(1),  # (B, 192, 1, 1)
        )

        self.projection = nn.Linear(192, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class PatchMatchTripletNetwork(nn.Module):
    """
    Triplet network using the enhanced PatchMatchEncoder.
    """
    def __init__(self, input_channels=1, embedding_dim=128):
        super(PatchMatchTripletNetwork, self).__init__()
        self.encoder = PatchMatchEncoder(input_channels, embedding_dim)

    def forward(self, anchor, positive, negative):
        anchor_embed = self.encoder(anchor)
        positive_embed = self.encoder(positive)
        negative_embed = self.encoder(negative)
        return anchor_embed, positive_embed, negative_embed


class TripletLoss(nn.Module):
    """
    Triplet margin loss with adjustable margin.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor_embed, positive_embed, negative_embed):
        return self.loss_fn(anchor_embed, positive_embed, negative_embed)
