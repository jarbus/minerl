#TODO deep q networks from demonstrations
import sys
import numpy as np
import gym
from tqdm import tqdm
import minerl
import torch
import argparse
from collections import Counter, namedtuple
import torch.nn as nn
import torch.nn.functional as F
from model import Model
import utils
from utils import *
from utils import ReplayBuffer, Experience
from losses import *
from train_loop import train, n_step_episode



LR = 0.001
SEQ_LEN = 1
BATCH_SIZE = 128
N = 10
GAMMA = 0.99
BUFFER_SIZE = 10000
MAX_EP_LEN = 500
TRAIN_STEPS = 5000
PRE_TRAIN_STEPS = 500
TAU=50

EPS_GREEDY=0.01
EPS_DEMO = 1.0
EPS_AGENT = 0.001

MODEL_PATH = "models/model.pt"
parser = argparse.ArgumentParser()
parser.add_argument("--train",action="store_true",help="Trains new model before evaluation")
parser.add_argument("--task",type=int,help="Specify task action space, 1-4", default=1)
args = parser.parse_args()

task = args.task


"""Create action masks.
An action mask is a (batch_size,11) vector whose values
are 0 for action indicies in the task and -inf otherwise.
We can add actions by this mask to ignore actions
not in our task.
"""
print(f"Creating action masks for task {task}:")
print(f"Actions allowed: {TASK_ACTIONS[task]}")
ENV_MASK = torch.full((1,11),0.0)
for a in TASK_ACTIONS[task]:
    ENV_MASK[:,a] = 1.0

print("ENV_MASK",ENV_MASK)

RawExp = namedtuple("RawExp", ["state", "action", "reward", "next_state","done"])

data = minerl.data.make("MineRLNavigateDense-v0")


behavior_network = Model()
target_network = Model()
if not args.train:
    behavior_network.load_state_dict(torch.load(MODEL_PATH))
    target_network.load_state_dict(torch.load(MODEL_PATH))

n_step_buffer = n_step_episode(N)
replay_buffer = ReplayBuffer(BUFFER_SIZE, epsilon_a=EPS_AGENT, epsilon_d=EPS_DEMO)

print("Preloading...")
idx = 0
step = 0
for s, a, r, sp, d in tqdm(data.batch_iter(
    num_epochs=1, seq_len=1, batch_size=1)):

    idx = idx + 1
    step = step + 1
    # Cap episode length
    if step > MAX_EP_LEN and not d:
        continue
    if d:
        step = 0
    # Cap amount of demo data to third buffer size
    if idx > BUFFER_SIZE / 3:
        break
    state = Navigatev0_obs_to_tensor(s)
    state_prime = Navigatev0_obs_to_tensor(sp)
    actions = Navigatev0_action_to_tensor(a,task=task).reshape((-1,11))

    # Load demo data into replay buffer with 0 TD error to start
    rawexp = RawExp(state, actions, torch.tensor(r[0][0],dtype=torch.float32), state_prime, d)
    for exp in n_step_buffer.setup_n_step_tuple(rawexp, is_demo=True):
        # Compute initial TD error

        td = abs(calcTD([exp], behavior_network,target_network,mask=ENV_MASK,n=N,gamma=GAMMA)[0].item()) + EPS_DEMO
        exp = exp._replace(td_error=td)
        replay_buffer.add(exp)


print("Done.")


model = train(replay_buffer,
              behavior_network,
              target_network,
              ENV_MASK,
              task=task,
              batch_size=BATCH_SIZE,
              pre_train_steps=PRE_TRAIN_STEPS,
              num_iters=TRAIN_STEPS,
              lr=LR,
              n=N,
              tau=TAU,
              path=MODEL_PATH,
              gamma=GAMMA,
              eps_greedy=EPS_GREEDY,
              eps_demo=EPS_DEMO,
              eps_agent=EPS_AGENT,
              max_ep_len=MAX_EP_LEN)
