#TODO deep q networks from demonstrations
import sys
import numpy as np
import gym
import minerl
import torch
import argparse
from collections import Counter, namedtuple
from torch import nn
from model import Model
import utils
from utils import *


LR = 0.001
SEQ_LEN = 1
BATCH_SIZE = 64
BUFFER_SIZE = 1000

MODEL_PATH = "models/model.pt"
parser = argparse.ArgumentParser()
parser.add_argument("--train",action="store_true",help="Trains new model before evaluation")
parser.add_argument("--task",type=int,help="Specify task action space, 1-4")
args = parser.parse_args()

TASK = args.task

# We multiply all actions not used for our task by 0
# We do this by multiplying output vectors by zero
# So weights for a specific action do not get updated

TRAINING_MASK = torch.zeros((BATCH_SIZE,11))
ENV_MASK = torch.zeros((1,11))
for a in TASK_ACTIONS[TASK]:
    TRAINING_MASK[:,a] = 1.0
    ENV_MASK[:,a] = 1.0


class DQNAgent():
    def __init__(self,
                BATCH_SIZE = 128,
                LEARNING_RATE = 0.0001,
                GAMMA = 0.999,
                EPSILON_START = 0.9,
                EPSILON_END = 0.0001,
                EPSILON_DECAY = 200,
                TARGET_UPDATE_RATE = 10):
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.tau = TARGET_UPDATE_RATE

        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


        # epsilon probaility for epsilon-greedy algorithm
        # epsilon starts with EPSILON_START and exponentially decays
        # at the rate of EPSILON_DECAY until it reaches to the 
        # value EPSILON_END 
        self.epsilon_start = EPSILON_START
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

        # create policy model and target model
        self.behavior_net = None
        self.target_net = None

    def update_target_model(self):
        '''
        updates the target network parameters to match with policy network
        '''
        self.target_net.load_state_dict(self.behavior_net.state_dict(), strict=True)
        pass

    def select_action(self, state, action_set):
        '''
        selects an action based on epsilon-greedy algorithm.
        In fact, with the chance of epsilon, the algorithm, with uniform distribution,
        randomly selects an action and with the (1-epsilon) chance it picks an action
        from the policy model.
        '''
        sample = np.random.rand()
        self.epsilon -= (self.epsilon_start - self.epsilon_end) / self.epsilon_decay

        if sample > self.epsilon:
            # the model returns Q value per action for a batch of input states
            # with shape (batch_size, n_actions). We pick the action 
            q = self.behavior_net(state)
            action_idx = np.argmax(q)

        else:
            action_idx = np.random.randint(0,12)
        return action_idx

    def optimize_model(self, sample_batch):
        '''
        arguments:
            sample_batch: list
                a list of random experiences from the replay memory.

        returns:
        '''
        
        # to split and concatenate "state", "action", "reward", "next_state", "done"
        # in separate lists
        batch = self.experience(*zip(*sample_batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # given a state s_t to the behavior network, compute Q(s_t)
        # then we use it to calucate Q(s_t, a_t) according to a greedy policy
        state_action_value_batch = self.behavior_net(state_batch).gather(1, action_batch)

        # compute V(s_t+1) = max_a(A(s_t+1, a)) for all next states 
        next_state_value_batch = torch.zeros(self.batch_size)
        # the target network updates the non-terminal state-values
        next_state_value_batch[done_batch] = self.target_net(state_batch[done_batch])
        #to compute the expected Q-values
        expected_state_action_value_batch = reward_batch + \
                                            next_state_value_batch * self.gamma
        
        # compute loss function



########################
# TRAIN ON EXPERT DATA #
########################
if args.train:
    model = Model()
    target_model = Model()
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
