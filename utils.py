from typing import *
from functools import partial
import torch
import numpy as np
import random
from collections import OrderedDict, namedtuple, deque

from pdb import set_trace as T

CAMERA_OPTIONS = torch.tensor([[0,-10],
                    [0,10],
                    [-10.0, 0.0],
                    [10.0,0.0]])
TASK_ACTIONS = {
        1: {0, 1, 10},
        2: {0, 1, 8,9,10},
        3: {0,1,2,3,7,8,9,10},
        4: {0,1,2,3,4,5,6,7,8,9,10}
}


Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done", "n_step_return", "state_tn", "is_demo", "td_error","is_bootstrapped"])

# Shorthand to put sequences and batches on same axis
def expand(tensor):
    tensor = torch.tensor(tensor)
    if len(tensor.size()) > 2:
        return torch.reshape(
            tensor, (tensor.size(0) * tensor.size(1),) + tensor.size()[2:]
        )
    else:
        return torch.reshape(tensor, (tensor.size(0) * tensor.size(1),))


# Shorthand to put first axis into batch_size rows of seq_len
def contract(tensor, batch_size, seq_len):
    assert batch_size * seq_len == tensor.size(0)
    return torch.reshape(tensor, (batch_size, seq_len) + tensor.size()[2:])


def Navigatev0_obs_to_tensor(obs: OrderedDict):
    """
    Parameters:
    -----------
        Observation:
              Dict({
                  "compassAngle": "Box(low=-180.0, high=180.0, shape=())",
                  "inventory": {
                          "dirt": "Box(low=0, high=2304, shape=())"
                  },
                  "pov": "Box(low=0, high=255, shape=(64, 64, 3))"
              })
    Returns:
    --------
        Pair of tensors, (POV, features)

        POV: (3,64,64) tensor
        features: (compassAngle, dirt count)

    """

    pov = torch.tensor(obs["pov"], dtype=torch.float)
    feats = torch.tensor(
            np.stack([obs["compassAngle"]/180.0, obs["inventory"]["dirt"]/10.0], axis=-1),
            dtype=torch.float)

    # Move channels into proper spot
    # (b,s,64,64,3) -> (b,s,3,64,64)
    pov = torch.transpose(pov, -3, -1)
    # Turn into batch for PyTorch model processing
    # (b,s,3,64,64) -> (b*s,64,64,3)
    pov = pov.reshape((-1,) + pov.size()[-3:])
    feats = feats.reshape((1,2))

    return pov, feats

def Navigatev0_action_to_tensor(act: OrderedDict, task=1):
    """
    Creates the following (batch_size, seq_len, 11) action tensor from Navigatev0 actions:

      0. cam left
      1. cam right
      2. cam up
      3. cam down
      4. place + jump
      5. place
      6. forward + attack
      7. attack
      8. forward + jump
      9. jump
      10. forward
    """

    batch_size, seq_len = act["jump"].shape
    PLACE_OPTIONS = {"none": 0, "dirt": 1}
    # ONE_HOT = {0: np.array([1, 0]), 1: np.array([0, 1])}
    out = torch.zeros((batch_size,seq_len,11))

    for b in range(batch_size):
        for s in range(seq_len):
            c = act["camera"]
            # We don't need to check if 0, 1, and 10 are in task actions
            # since they always will be
            task_acts = TASK_ACTIONS[task]

            # Set camera left
            if c[b,s][0] < -10 and abs(c[b,s][0]) >= abs(c[b,s][1]):
                out[b,s][0] = 1
            # Set camera right
            elif c[b,s][0] > 10 and abs(c[b,s][0]) >= abs(c[b,s][1]):
                out[b,s][1] = 1
            # Set camera up
            elif 2 in task_acts and c[b,s][1] < -10 and abs(c[b,s][1]) >= abs(c[b,s][0]):
                out[b,s][2] = 1
            elif 3 in task_acts and c[b,s][1] > 10 and abs(c[b,s][1]) >= abs(c[b,s][0]):
                out[b,s][3] = 1
            elif PLACE_OPTIONS[act["place"][b,s]] == 1:
                if 4 in task_acts and act["jump"][b,s] == 1:
                    out[b,s][4] = 1
                elif 5 in task_acts:
                    out[b,s][5] = 1
            elif act["attack"][b,s] == 1:
                if 6 in task_acts and act["forward"][b,s] == 1:
                    out[b,s][6] = 1
                elif 7 in task_acts:
                    out[b,s][7] = 1
            elif act["jump"][b,s] == 1:
                if 8 in task_acts and act["forward"][b,s] == 1:
                    out[b,s][8] = 1
                elif 9 in task_acts:
                    out[b,s][9] = 1
            else:
                out[b,s][10] = 1

    return out

index_to_actions = {
        0: np.array([0,-10]),
        1: np.array([0,10]),
        2: np.array([-10,0]),
        3: np.array([10,0]),
        4: ["place","jump"],
        5: ["place"],
        6: ["forward","attack"],
        7: ["attack"],
        8: ["forward","jump"],
        9: ["jump"],
        10: ["forward"]}

def action_tensor_to_Navigatev0(action_vec_torch, epsilon=0.01, evaluation=False, task=1):
    # Constructs a one-hot action dict for stepping through the environment

    # Task parameter must be in range [1, 4]
    action_vec = action_vec_torch.detach().reshape((11,))
    action_dict = OrderedDict()
    action_dict["attack"] = 0
    action_dict["camera"] = np.zeros(2,dtype=np.int8)
    action_dict["forward"] = 0
    action_dict["jump"] = 0
    action_dict["place"] = 0
    action_dict["back"] = 0
    action_dict["right"] = 0
    action_dict["left"] = 0
    action_dict["sneak"] = 0
    action_dict["sprint"] = 0


    # Only look at the actions we care about for the task
    actions = TASK_ACTIONS[task]

    # Choose the best action from available actions
    if random.random() < epsilon:
        max_idx = random.choice(list(actions))
    else:
        max_idx = max(((idx,action_vec[idx]) for idx in actions),\
                    key=lambda x:x[1])[0]
    if max_idx not in actions:
        print("wrong action selected:")
        print(max_idx)
    if max_idx < 4:
        action_dict["camera"] = index_to_actions[max_idx]
    else:
        for action in index_to_actions[max_idx]:
            action_dict[action] = 1

    return action_dict



class ReplayBuffer():
    # Note max size is currently max size of the buffer only, so it is possible
    #   that the length of a replay buffer (as gotten through len()) is larger
    #   than its max size
    def __init__(self, max_size, epsilon_a=0.001, epsilon_d=1.0) :
        self.max_size   = max_size
        self.buffer     = deque(maxlen=max_size)    # Stores sim data
        self.reserve    = list()                    # Stores demo data
        #self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done", "n_step_return", "state_tn", "is_demo", "td_error"])
        self.eps_a      = epsilon_a
        self.eps_d      = epsilon_d

    def __len__(self) :
        return len(self.buffer) + len(self.reserve)

    # Allows subscripting for get - note that there is no guarantee the nth
    #   element remains the nth element after any operation
    # The reserve data (demonstration data) comes before the sim data
    def __getitem__(self, i) :
        if isinstance(i, int) or isinstance(i, np.int64):
            if i < 0 :      # Handle negative indices
                i += len(self)

            if i >= 0 and i < len(self.reserve) :
                return self.reserve[i]
            elif i >= len(self.reserve) and i < len(self) :
                return self.buffer[i-len(self.reserve)]
            else :
                raise IndexError(f"Index {i} is out of bounds.")
        elif isinstance(i, slice) :
            start, stop, step = i.indices(len(self))
            return [self[j] for j in range(start, stop, step)]
        else :
            raise TypeError(f"Invalid argument type: {type(i)}")


    def __setitem__(self, i, val) :
        if isinstance(i, int) or isinstance(i, np.int64):
            if i < 0 :      # Handle negative indices
                i += len(self)

            if i >= 0 and i < len(self.reserve) :
                self.reserve[i] = val
            elif i >= len(self.reserve) and i < len(self) :
                self.buffer[i-len(self.reserve)] = val
            else :
                raise IndexError(f"Index {i} is out of bounds.")
        else :
            raise TypeError(f"Invalid argument type: {type(i)}")

    # Takes state, action, reward, next state, done, n step return, state t+n,
    #   and is_demo
    # def add(self, s, a, r, sp, d, nsr, stn, i_d, td) :
    #     exp = Experience(s, a, r, sp, d, nsr, stn, i_d, td)
    def add(self, exp: Experience):
        if exp.is_demo:
            self.reserve.append(exp)
        else :
            self.buffer.append(exp)

    # Takes list of indices, list of td errors, both same length
    # Get list of indices from sample()
    def update_td(self, indices, new_errors) :
        assert(len(indices) == len(new_errors))
        for i in range(len(indices)) :
            self[indices[i]] = self[indices[i]]._replace(td_error=new_errors[i])

    # Takes int sample_size, returns list of sample_size named tuples randomly
    #   selected from the sim and demo data
    # See above for named elements of the tuples
    # INDICES ONLY VALID UNTIL ADD CALLED
    # MAKE SURE TO CALL update_td WITH INDICES AND NEW TD ERRORS
    def sample(self, sample_size) :
        p_sum = sum([e.td_error for e in self.reserve]) + sum(e.td_error for e in self.buffer) + self.eps_d * len(self.reserve) + self.eps_a * len(self.buffer)
        probs = []
        if len(self.reserve) > 0 :
            probs += [(e.td_error + self.eps_d)/p_sum for e in self.reserve]
        if len(self.buffer) > 0 :
            probs += [(e.td_error + self.eps_a)/p_sum for e in self.buffer]
        indices = np.random.choice(len(self), size=sample_size, replace=False, p=probs)
        # TODO this is terrible... at least it's only n additions, but...
        return [self[i.astype(int)] for i in indices], indices, [probs[i.astype(int)] for i in indices], p_sum
