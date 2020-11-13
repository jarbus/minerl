import sys
import gym
import minerl
import torch
from torch import nn
from model import Model
from utils import *

LR = 0.001
SEQ_LEN = 10
BATCH_SIZE = 64
# Sample some data from the dataset!

model = Model(2, 200)

cross_ent = nn.CrossEntropyLoss()
mse = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#old_loss = float("Inf")
#data = minerl.data.make("MineRLNavigate-v0")
# Iterate through a single epoch using sequences of at most 32 steps
#it = 0
#for s, a, r, sp, d in data.batch_iter(
#    num_epochs=1, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
#):
#    # optimizer changes params, loss computes the gradients
#    # dicts are arrays of samples
#    pov_tensor, feat_tensor = Navigatev0_obs_to_tensor(s)
#    pov_tensor, feat_tensor = (
#        torch.transpose(expand(pov_tensor), 1, 3),
#        expand(feat_tensor),
#    )
#    action_tensor = Navigatev0_action_to_tensor(a)
#
#    action_tensor = {a: expand(t) for a, t in action_tensor.items()}
#    #   4. Write a training loop
#    optimizer.zero_grad()
#    outputs = model(pov_tensor, feat_tensor)
#    loss = cross_ent(outputs[:, 0:2], action_tensor["attack"])
#    loss += cross_ent(outputs[:, 2:4], action_tensor["back"])
#    loss = mse(outputs[:, 4:6], action_tensor["camera"])
#    loss += cross_ent(outputs[:, 6:8], action_tensor["forward"])
#    loss += cross_ent(outputs[:, 8:10], action_tensor["jump"])
#    loss += cross_ent(outputs[:, 10:12], action_tensor["left"])
#    loss += cross_ent(outputs[:, 12:14], action_tensor["right"])
#    loss += cross_ent(outputs[:, 14:16], action_tensor["place"])
#    loss += cross_ent(outputs[:, 16:18], action_tensor["sneak"])
#    loss += cross_ent(outputs[:, 18:20], action_tensor["sprint"])
#    loss.backward()
#    optimizer.step()
#    it += 1
#    if it % 1 == 0:
#        print(f"Loss: {loss.item()}")
#    if it >= 1000:
#        # if loss.item() > old_loss + 0.5 or it >= 30:
#        print(f"Converged at iter {it} with loss {loss.item()}")
#        break
#    old_loss = loss.item()

env = gym.make("MineRLNavigate-v0")

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
