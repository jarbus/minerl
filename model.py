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
            nn.Conv2d(3, 16, 5, stride=2),
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
            nn.Linear(96, 50),
        )
        # Regularization layer
        self.l1 = nn.Linear(50 + 2, 50)
        self.r1 = nn.LeakyReLU()
        self.out = nn.Linear(50, 11)

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
        full_embed = self.l1(torch.cat((pov, feats), dim=1))
        full_embed = self.r1(full_embed)
        out        = self.out(full_embed)
        return out
