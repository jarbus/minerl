from typing import *
from functools import partial
import torch
import numpy as np
import random
from collections import OrderedDict, namedtuple, deque

from pdb import set_trace as T

CAMERA_OPTIONS = torch.tensor([[-90.0, 0.0],
                    [90.0,0.0],
                    [0.0, 0.0],
                    [0,-45],
                    [0,45]])

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


# distribution is of (batch_size, seq_len, distribution)
def sample(action_vector, epsilon=0.01,evaluation=False):
    sample = torch.zeros(9,dtype=torch.int8)
    if evaluation:
        sample[0] = round(action_vector[0].item())
        sample[torch.argmax(action_vector[1:6])] = 1
        sample[6] = round(action_vector[6].item())
        sample[7] = round(action_vector[7].item())
        sample[8] = round(action_vector[8].item())
    elif random.random() < epsilon:
        sample[0] = random.randint(0,1)
        sample[random.randint(1,5)] = 1
        sample[6] = random.randint(0,1)
        sample[7] = random.randint(0,1)
        sample[8] = random.randint(0,1)
    else:
        sample[0] = 1 if random.random() < action_vector[0] else 0
        camera_thresh = random.random()
        sum_ = 0
        for i in range(1,6):
            sum_ += action_vector[i]
            if sum_ < random.random():
                sample[i] = 1
                break
        sample[6] = 1 if random.random() < action_vector[6] else 6
        sample[7] = 1 if random.random() < action_vector[7] else 7
        sample[8] = 1 if random.random() < action_vector[8] else 8

    return sample

def batch_sample(array, evaluation=False):
    batch_size, seq_len, dist_size = array.shape
    # multithread sampling from probability dist with specified evaluation
    _sample = partial(sample, evaluation=evaluation)
    # sample from all timesteps in all batches by flattening timesteps
    samples = torch.tensor(
        map(_sample, my_array.reshape((batch_size + seq_len, dist_size)))
    )
    return torch.reshape(samples, (batch_size, seq_len, dist_size))


def Navigatev0_obs_to_tensor(obs: OrderedDict):
    """
    Observation Space:
        Dict({
            "compassAngle": "Box(low=-180.0, high=180.0, shape=())",
            "inventory": {
                    "dirt": "Box(low=0, high=2304, shape=())"
            },
            "pov": "Box(low=0, high=255, shape=(64, 64, 3))"
        })
    """
    return (
        torch.tensor(obs["pov"], dtype=torch.float),
        torch.tensor(
            np.stack([obs["compassAngle"], obs["inventory"]["dirt"]], axis=-1),
            dtype=torch.float,
        ),
    )


def Navigatev0_action_to_tensor(act: OrderedDict):
    # Convert batch player data to tensors

    batch_size, seq_len = act["jump"].shape
    PLACE_OPTIONS = {"none": 0, "dirt": 1}
    # ONE_HOT = {0: np.array([1, 0]), 1: np.array([0, 1])}
    out = torch.zeros((batch_size,seq_len,9))
    out[:,:,0] = torch.tensor(act["attack"])
    out[:,:,6] = torch.tensor(act["forward"])
    out[:,:,7] = torch.tensor(act["jump"])

    for b in range(batch_size):
        for s in range(seq_len):
            c = act["camera"]
            if c[b,s][0] < -10 and abs(c[b,s][0]) >= abs(c[b,s][1]):
                out[b,s][1] = 1
            elif c[b,s][0] > 10 and abs(c[b,s][0]) >= abs(c[b,s][1]):
                out[b,s][2] = 1
            elif c[b,s][1] < -10 and abs(c[b,s][1]) >= abs(c[b,s][0]):
                out[b,s][4] = 1
            elif c[b,s][1] > 10 and abs(c[b,s][1]) >= abs(c[b,s][0]):
                out[b,s][5] = 1
            else:
                out[b,s][3] =1
            out[b,s][8] = PLACE_OPTIONS[act["place"][b,s]]

    return out


def action_tensor_to_Navigatev0(action_vec_torch, epsilon=0.01, evaluation=False):
    # Constructs a one-hot action dict for stepping through the environment
    action_vec = action_vec_torch.detach()
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

    actions = sample(action_vec, epsilon=epsilon, evaluation=evaluation)
    action_dict["camera"] = CAMERA_OPTIONS[torch.argmax(actions[1:6])-1]
    for key, idx in zip(["attack","forward","jump","place"],  [0,6,7,8]):
        action_dict[key] = actions[idx].item()
    return action_dict

def batch_action_tensor_to_Navigatev0(action_vec_torch, epsilon=0.01, evaluation=False):
    # Used to convert q-value estimates to dictionaries for compute in loss function
    b_size = action_vec_torch.size(0)
    action_vec = action_vec_torch.detach()
    action_dict = OrderedDict()
    for key in ["attack", "forward","jump","place","back","right","left","sneak""sprint"]:
        action_dict[key] = np.zeros((b_size))
    action_dict["camera"] = np.zeros((b_size,2))

    for i,key in enumerate(["attack","camera","forward","jump","place"]):
        for b in range(b_size):
            if i == 0:
                action_dict[key][b] = action_vec[b][i]
            # Make camera action a weighted sum
            elif i == 1:
                for j in range(1,1+len(CAMERA_OPTIONS)):
                    action_dict["camera"][b] += action_vec[b][j] * CAMERA_OPTIONS[j]
            else:
                action_dict[key][b] = action_vec[b][i+4]
    return action_dict


class ReplayBuffer :
    def __init__(self, max_size, seed=None) :
        self.max_size   = max_size
        self.buffer     = deque(maxlen=max_size)
        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
        self.seed       = random.seed(seed)

    def add(self, s, a, r, sp, d) :
        self.buffer.append(self.experience(s, a, r, sp, d))

    def sample(self, sample_size) :
        return random.sample(self.buffer, sample_size)


if __name__ == "__main__":
    print("sample tests")
    dist = [0.25, 0.75]
    counts = np.zeros(2)
    for i in range(100):
        counts += sample(dist, False)
    print(counts)
