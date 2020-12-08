import sys
import numpy as np
import gym
import minerl
import torch
import argparse
from collections import Counter, namedtuple
import torch.nn as nn
import torch.nn.functional as F
from model import Model
import utils
from utils import *
from losses import *

# Test J_L2 Loss
m = Model()
print(J_L2(m))

# Test J_E loss
pred = torch.arange(9).reshape((3,3))
demo = torch.tensor([[0,0,0],[0,1,0],[1,0,0]])
margin = 1
pred + margin*demo
x,_ = torch.max(pred + margin*demo,dim=1)
y = torch.sum(pred * demo,dim=1)
print(J_E(pred, demo))
