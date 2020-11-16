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
# Sample some data from the dataset!
parser = argparse.ArgumentParser()
parser.add_argument("--train",action="store_true",help="Trains new model before evaluation")
args = parser.parse_args()

if args.train:
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
    output_history=Counter()

    for s, a, r, sp, d in data.batch_iter(
        num_epochs=1, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
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
        for i in range(outputs.size(0)):
            output_history[torch.argmax(outputs[i]).item()] += 1

        loss = mlsml(outputs, action_tensor)
        if it % 100 == 0:
            print(f"Iteration {it} Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        it += 1
        if it > 200:
            break

    torch.save(model.state_dict(),MODEL_PATH)
    print("Training counter:",output_history)

else:
    model = Model()
    model = model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

env = gym.make("MineRLNavigateDense-v0")

obs = env.reset()
done = False
net_reward = 0

it = 0
output_history=Counter()
while not done:

    pov, feats = Navigatev0_obs_to_tensor(obs)
    # Move channels into proper spot
    pov = torch.transpose(pov, 0, 2)
    # Turn into batch
    pov = pov.expand((1,) + pov.size())
    feats = feats.expand((1,) + feats.size())

    action_tensor = model.forward(pov, feats)
    output_history[torch.argmax(action_tensor).item()] += 1
    action_tensor[:,3] = 0.0
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
print("Evaluation counter:",output_history)
