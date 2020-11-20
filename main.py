#TODO deep q networks from demonstrations
import sys
import gym
import minerl
import torch
import argparse
from collections import Counter
from torch import nn
from model import Model
from utils import *

LR = 0.001
SEQ_LEN = 1
BATCH_SIZE = 64
MODEL_PATH = "models/model.pt"
parser = argparse.ArgumentParser()
parser.add_argument("--train",action="store_true",help="Trains new model before evaluation")
args = parser.parse_args()

########################
# TRAIN ON EXPERT DATA #
########################
if args.train:
    model = Model()
    bce = nn.BCELoss()
    cross_ent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    mlsml = nn.MultiLabelSoftMarginLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    old_loss = float("Inf")
    data = minerl.data.make("MineRLNavigateDense-v0")
    # Iterate through a single epoch using sequences of at most 32 steps
    it = 0
    output_history=Counter()

    for s, a, r, sp, d in data.batch_iter(
        num_epochs=1, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
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
        for i in range(outputs.size(0)):
            output_history[torch.argmax(outputs[i]).item()] += 1

        loss = bce(outputs[:,0], action_tensor[:,0])
        loss += cross_ent(outputs[:,1:6], torch.argmax(action_tensor[:,1:6],dim=1))
        loss += bce(outputs[:,6], action_tensor[:,6])
        loss += bce(outputs[:,7], action_tensor[:,7])
        loss += bce(outputs[:,8], action_tensor[:,8])
        if it % 100 == 0:
            print(f"Iteration {it} Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        it += 1
        if it > 200:
            break

    torch.save(model.state_dict(),MODEL_PATH)
    print("Training counter:",output_history)

# Loads previously trained model
else:
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


################
# SIMULATION
################

env = gym.make("MineRLNavigateDense-v0")

obs = env.reset()
done = False
net_reward = 0
it = 0

# Counts actions taken
# Doesn't affect training
output_history=Counter()

# Evaluates on environment
while not done:

    pov, feats = Navigatev0_obs_to_tensor(obs)
    # Move channels into proper spot
    # (64,64,3) -> (3,64,64)
    pov = torch.transpose(pov, 0, 2)
    # Turn into batch
    pov = pov.expand((1,) + pov.size())
    feats = feats.expand((1,) + feats.size())

    action_tensor = model(pov, feats)
    output_history[torch.argmax(action_tensor).item()] += 1

    # Add fake sequence dimension
    # action_tensor = action_tensor.expand((1,) + action_tensor.size())

    action_dict = action_tensor_to_Navigatev0(action_tensor[0], evaluation=True)

    obs, reward, done, info = env.step(action_dict)
    net_reward += reward
    if net_reward > 0:
        print(f"{net_reward=} at {it=}")
    it += 1
    if it > 1000:
        break
print("Total reward: ", net_reward)
print("Evaluation counter:",output_history)
