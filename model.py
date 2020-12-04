import numpy as np
import torch
import torch.nn as nn
from utils import *
from collections import OrderedDict
from torchsummary import summary


class Model(nn.Module):
    """Example usage:

    model = Model()
    outputs = model(pov_tensor, feat_tensor)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Convolutional network architecture
        self.image_embed = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(True),
            nn.BatchNorm2d(24),
            nn.Flatten(),
            nn.Linear(864, 100),
        )
        # Regularization layer
        self.dropout = nn.Dropout(p=0.05)
        self.l1 = nn.Linear(100 + 2, 200)
        self.r1 = nn.LeakyReLU()
        self.l2 = nn.Linear(200, 100)
        self.r2 = nn.LeakyReLU()
        self.out = nn.Linear(100, 11)
        # The following allows use to sample each sub-action
        # from its own distribution

    """Model to approximate Q values.

    Input
    -----
        pov:        (batch_size, 3, 64, 64) tensor of player view
        input_size: (batch_size, 2)

    Returns
    -------
        action:     (batch_size, 9) tensor with indicies:
            0: attack probability
            1-5: CAMERA_OPTIONS[0-4]
            6: forward probability
            7: jump probability
            8: place probability

    """
    def forward(self, pov, feats):
        pov = self.image_embed(pov)
        dropout = self.dropout(torch.cat((pov, feats), dim=1))
        full_embed = self.l1(dropout)
        full_embed = self.r1(full_embed)
        full_embed = self.l2(full_embed)
        full_embed = self.r2(full_embed)
        out        = self.out(full_embed)
        return out
