import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMatchEncoder(nn.Module):
    """
    Enhanced encoder with increased capacity using depthwise separable convolutions
    and larger feature maps for improved generalization.
    """
    def __init__(self, input_channels=1, embedding_dim=128):
        super(PatchMatchEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # Increased channels
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)

class PatchMatchTripletNetwork(nn.Module):
    """
    Triplet network using the enhanced PatchMatchEncoder.
    """
    def __init__(self, input_channels=1, embedding_dim=128):  # Fixed default
        super(PatchMatchTripletNetwork, self).__init__()
        self.encoder = PatchMatchEncoder(input_channels, embedding_dim)

    def forward(self, anchor, positive, negative):
        anchor_embed = self.encoder(anchor)
        positive_embed = self.encoder(positive)
        negative_embed = self.encoder(negative)
        return anchor_embed, positive_embed, negative_embed

class TripletLoss(nn.Module):
    """
    Standard triplet margin loss.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor_embed, positive_embed, negative_embed):
        return self.loss_fn(anchor_embed, positive_embed, negative_embed)
