from typing import *
from functools import partial
import torch
import numpy as np
import random

from pdb import set_trace as T

CAMERA_OPTIONS = np.array([[-90.0, 0.0],
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
def sample(distribution, epsilon=0.01,evaluation=False):

    if evaluation or random.random()<epsilon:
        return torch.argmax(distribution)
    return random.randint(0,len(distribution)-1)
    #distribution = torch.nn.Softmax(dim=-1)(distribution)
    #sum_so_far = 0
    #r = random.random()
    #for idx, prob in enumerate(distribution):
    #    sum_so_far += prob
    #    if r <= sum_so_far:
    #        return idx


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

    batch_size, seq_len = act["jump"].shape
    PLACE_OPTIONS = {"none": 0, "dirt": 1}
    # ONE_HOT = {0: np.array([1, 0]), 1: np.array([0, 1])}

    options = np.array([[-90.0, 0.0],
                        [90.0,0.0],
                        [0.0, 0.0],
                        [0,-45],
                        [0,45]])

    place = np.zeros((batch_size, seq_len, 2), dtype=int)
    attack = np.zeros((batch_size, seq_len, 2), dtype=int)
    forward = np.zeros((batch_size, seq_len, 2), dtype=int)
    jump = np.zeros((batch_size, seq_len, 2), dtype=int)
    camera = np.zeros((batch_size, seq_len,5), dtype=int)
    for b in range(batch_size):
        for s in range(seq_len):
            c = act["camera"]
            if c[0] < -10 and abs(c[0]) >= abs(c[1]):
                camera[b,s] = options[0]
            elif c[0] > 10 and abs(c[0]) >= abs(c[1]):
                camera[b,s] = options[1]
            elif c[1] < -10 and abs(c[1]) >= abs(c[0]):
                camera[b,s] = options[3]
            elif c[1] > 10 and abs(c[1]) >= abs(c[0]):
                camera[b,s] = options[4]
            else:
                camera[b,s] = options[2]
            place[b, s] = PLACE_OPTIONS[act["place"][b, s]]

    act["camera"] = camera
    act["place"] = place
    return act


def action_tensor_to_Navigatev0(action_vec_torch, epsilon=0.01, evaluation=False):
    """
    Dict({
        "attack": "Discrete(2)",
        "camera": "Box(low=-180.0, high=180.0, shape=(2,))",
        "forward": "Discrete(2)",
        "jump": "Discrete(2)",
        "place": "Enum(dirt,none)",
    })
    """
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

    idx = sample(action_vec, epsilon=epsilon, evaluation=evaluation)
    if 1 <= idx <= 5:
        action_dict["camera"] = CAMERA_OPTIONS[idx-1]
    else:
        action_dict[["attack",0,0,0,0,0,"forward","jump","place"][idx]] = 1
    return action_dict


if __name__ == "__main__":
    print("sample tests")
    dist = [0.25, 0.75]
    counts = np.zeros(2)
    for i in range(100):
        counts += sample(dist, False)
    print(counts)
