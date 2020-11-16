import sys
import gym
import minerl
import torch
from torch import nn
from model import Model
from utils import *

LR = 0.001
SEQ_LEN = 1
BATCH_SIZE = 64
# Sample some data from the dataset!

model = Model()

cross_ent = nn.CrossEntropyLoss()
mse = nn.MSELoss()
mlsml = nn.MultiLabelSoftMarginLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

old_loss = float("Inf")
data = minerl.data.make("MineRLNavigateDense-v0")
# Iterate through a single epoch using sequences of at most 32 steps
it = 0
for s, a, r, sp, d in data.batch_iter(
    num_epochs=3, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
):
    # converts state dictionary to tensor information
    pov_tensor, feat_tensor = Navigatev0_obs_to_tensor(s)
    pov_tensor, feat_tensor = (
        torch.transpose(expand(pov_tensor), 1, 3),
        expand(feat_tensor),
    )
    # Convert action dict to tensor and put batches & sequences on same axis
    action_tensor = Navigatev0_action_to_tensor(a)
    action_tensor = expand(action_tensor)

    outputs = model(pov_tensor, feat_tensor)

    loss = mlsml(outputs, action_tensor)
    loss.backward()
    optimizer.step()
    it += 1

env = gym.make("MineRLNavigateDense-v0")

obs = env.reset()
done = False
net_reward = 0

it = 0
while not done:

    pov, feats = Navigatev0_obs_to_tensor(obs)
    # Move channels into proper spot
    pov = torch.transpose(pov, 0, 2)
    # Turn into batch
    pov = pov.expand((1,) + pov.size())
    feats = feats.expand((1,) + feats.size())

    action_tensor = model.forward(pov, feats)
    # Add fake sequence
    action_tensor = action_tensor.expand((1,) + action_tensor.size())

    action_dict = action_tensor_to_Navigatev0(action_tensor, evaluation=True)
    obs, reward, done, info = env.step(action_dict)
    net_reward += reward
    if net_reward > 0:
        print(f"{net_reward=} at {it=}")
    it += 1
    if it > 1000:
        break
print("Total reward: ", net_reward)
