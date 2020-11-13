import numpy as np
import torch
import torch.nn as nn
from utils import *
from collections import OrderedDict
from torchsummary import summary


class Model(nn.Module):
    def __init__(self, input_size, output_size):
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
        self.feat_embed = nn.Linear(input_size, 20)
        self.dropout = nn.Dropout(p=0.05)
        self.l1 = nn.Linear(100 + 20, 200)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(200, output_size)
        self.r2 = nn.ReLU()

        self.out = nn.Linear(output_size, 9)

        # self.classifier = nn.Conv2d(128, 10, 1)
        # self.avgpool = nn.AvgPool2d(6, 6)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, pov, inputs):
        pov = self.image_embed(pov)
        feats = self.feat_embed(inputs)
        dropout = self.dropout(torch.cat((pov, feats), dim=1))
        full_embed = self.l1(dropout)
        full_embed = self.r1(full_embed)
        full_embed = self.l2(full_embed)
        full_embed = self.r2(full_embed)
        return self.out(full_embed)

    def sample(self, pov, feats, evaluation=False):
        pov = torch.reshape(pov, (1,) + pov.size())
        feats = torch.reshape(feats, (1,) + feats.size())
        outputs = self.forward(pov, feats)
        act = OrderedDict()
        act["attack"] = sample(outputs[0, 0:2], evaluation=evaluation)
        act["back"] = sample(outputs[0, 2:4], evaluation=evaluation)
        act["camera"] = outputs[0, 4:6].detach().numpy()
        act["forward"] = sample(outputs[0, 6:8], evaluation=evaluation)
        act["jump"] = sample(outputs[0, 8:10], evaluation=evaluation)
        act["left"] = sample(outputs[0, 10:12], evaluation=evaluation)
        act["right"] = sample(outputs[0, 12:14], evaluation=evaluation)
        act["place"] = sample(outputs[0, 14:16], evaluation=evaluation)
        act["sneak"] = sample(outputs[0, 16:18], evaluation=evaluation)
        act["sprint"] = sample(outputs[0, 18:20], evaluation=evaluation)

        return act


if __name__ == "__main__":
    model = Model(2, 200)
    summary(model, [(3, 64, 64), (2,)])
