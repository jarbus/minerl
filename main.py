#TODO deep q networks from demonstrations
import sys
import gym
import minerl
import torch
import argparse
from collections import Counter
from torch import nn
from model import Model
import utils
from utils import *


LR = 0.001
SEQ_LEN = 1
BATCH_SIZE = 64
BUFFER_SIZE = 1000
TASK = 1

# We multiply all actions not used for our task by 0
# We do this by multiplying output vectors by zero
# So weights for a specific action do not get updated

TRAINING_MASK = torch.zeros((BATCH_SIZE,11))
ENV_MASK = torch.zeros((1,11))
for a in TASK_ACTIONS[TASK]:
    TRAINING_MASK[:,a] = 1.0
    ENV_MASK[:,a] = 1.0

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
        action_tensor = Navigatev0_action_to_tensor(a,task=TASK)
        action_tensor = expand(action_tensor)

        outputs = model(pov_tensor, feat_tensor)
        outputs = outputs * TRAINING_MASK
        # Count agent actions
        for i in range(outputs.size(0)):
            act = torch.argmax(outputs[i]).item()
            # Sometimes all valid actions will have negative values, if so just choose 10, "forward"
            act = 10 if outputs[i][act] == 0 else act
            output_history[act] += 1
        # Count number of demonstrator actions
        #for i in range(action_tensor.size(0)):
        #    output_history[torch.argmax(action_tensor[i]).item()] += 1

        # Cross Entropy loss for camera action distribution
        loss = cross_ent(outputs, torch.argmax(action_tensor,dim=1))
        if it % 50 == 0:
            print(f"Iteration {it} Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        it += 1
        if it > 400:
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

mem = ReplayBuffer(BUFFER_SIZE)

output_history=Counter()
# Evaluates on environment
while not done:

    # Convert environment observations to PyTorch tensors
    pov, feats = Navigatev0_obs_to_tensor(obs)
    # Move channels into proper spot
    # (64,64,3) -> (3,64,64)
    pov = torch.transpose(pov, 0, 2)
    # Turn into batch for PyTorch model processing
    # (1,64,64,3)
    pov = pov.expand((1,) + pov.size())
    feats = feats.expand((1,) + feats.size())

    # Compute actions
    outputs = model(pov, feats)
    outputs = outputs * ENV_MASK

    # Count agent actions
    for i in range(outputs.size(0)):
        act = torch.argmax(outputs[i]).item()
        # Sometimes all valid actions will have negative values, if so just choose 10, "forward"
        output_history[act] += 1
    # Turn action tensors into valid Minecraft actions
    action_dict = action_tensor_to_Navigatev0(outputs[0], evaluation=True, task=TASK)
    # Perform action in Minecraft
    obs, reward, done, info = env.step(action_dict)

    state = (pov, feats)
    # TODO fix this.. cop out: prev_state doesnt exist for first, doing calculations
    #   for state twice is wasted time
    try:
        mem.add(prev_state, action_dict, reward, state, done)
    except: pass
    prev_state = state

    net_reward += reward
    if net_reward > 0:
        print(f"{net_reward} at {it}")
    it += 1
    if it > 100:
        print("Total reward: ", net_reward)
        print("Evaluation counter:",output_history)
        break
