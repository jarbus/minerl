import numpy as np
import torch
import torch.nn as nn
from utils import *
from collections import OrderedDict
from torchsummary import summary


class Model(nn.Module):
    """Model to approximate Q values.

    Input
    -----
        pov: batch_size x 3 x 64 x 64 POV
        input_size

    """
    def __init__(self):
        super(Model, self).__init__()
        self.image_embed = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 24, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(24),
            nn.Flatten(),
            nn.Linear(864, 100),
        )
        #self.feat_embed = nn.Linear(2, 20)
        self.dropout = nn.Dropout(p=0.05)
        self.l1 = nn.Linear(100 + 2, 200)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(200, 100)
        self.r2 = nn.ReLU()
        self.out = nn.Linear(100, 9)


    def forward(self, pov, feats):
        pov = self.image_embed(pov)
        dropout = self.dropout(torch.cat((pov, feats), dim=1))
        full_embed = self.l1(dropout)
        full_embed = self.r1(full_embed)
        full_embed = self.l2(full_embed)
        full_embed = self.r2(full_embed)
        return self.out(full_embed)


if __name__ == "__main__":
    model = Model(2, 200)
    summary(model, [(3, 64, 64), (2,)])
