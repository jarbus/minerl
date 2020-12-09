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
from random import randint

NUM_ACTIONS = 11

# Test J_L2 Loss
behavior_network = Model()
print(J_L2(behavior_network))

# Test J_E loss
pred = torch.arange(9).reshape((3,3))
demo = torch.tensor([[0,0,0],[0,1,0],[1,0,0]])
margin = 1
pred + margin*demo
x,_ = torch.max(pred + margin*demo,dim=1)
y = torch.sum(pred * demo,dim=1)
print(J_E(pred, demo))

target_network = Model()
def generate_dummy_states(batch_size):
    return torch.rand(batch_size,3,64,64), torch.rand(batch_size, 2)
def generate_dummy_is_demo(batch_size):
    is_demo = torch.zeros((batch_size, NUM_ACTIONS))
    for i in range(batch_size):
        is_demo[i,randint(0,1)] = 1
    return is_demo
samples = (generate_dummy_states(3),
            None,
            None,
            generate_dummy_is_demo(3),
            None)
loss = J_Q(target_network,
           behavior_network,
           samples)
print(loss)

