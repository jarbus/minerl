from typing import *
from functools import partial
import torch
import numpy as np
import random

from pdb import set_trace as T


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
def sample(distribution, evaluation=False):

    if evaluation:
        return torch.argmax(distribution)
    distribution = torch.nn.Softmax(dim=-1)(distribution)
    sum_so_far = 0
    r = random.random()
    for idx, prob in enumerate(distribution):
        sum_so_far += prob
        if r <= sum_so_far:
            return idx


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

    place = np.zeros((batch_size, seq_len), dtype=int)
    for b in range(batch_size):
        for s in range(seq_len):
            place[b, s] = PLACE_OPTIONS[act["place"][b, s]]
    act["place"] = place
    return act


def action_tensor_to_Navigatev0(action_vec_torch, evaluation=False):
    """
    Dict({
        "attack": "Discrete(2)",
        "back": "Discrete(2)",
        "camera": "Box(low=-180.0, high=180.0, shape=(2,))",
        "forward": "Discrete(2)",
        "jump": "Discrete(2)",
        "left": "Discrete(2)",
        "place": "Enum(dirt,none)",
        "right": "Discrete(2)",
        "sneak": "Discrete(2)",
        "sprint": "Discrete(2)"
    })
    """
    action_vec = action_vec_torch.detach().numpy()[0]
    action_dict = OrderedDict()
    action_dict["attack"] = batch_sample(action_vec[:, :, 0:2], evaluation)
    action_dict["back"] = batch_sample(action_vec[:, :, 2:4], evaluation)
    action_dict["camera"] = action_vec[:, :, 4:6]
    action_dict["forward"] = batch_sample(action_vec[:, :, 6:8], evaluation)
    action_dict["jump"] = batch_sample(action_vec[:, :, 8:10], evaluation)
    action_dict["left"] = batch_sample(action_vec[:, :, 10:12], evaluation)
    action_dict["place"] = batch_sample(action_vec[:, :, 12:14], evaluation)
    action_dict["right"] = batch_sample(action_vec[:, :, 14:16], evaluation)
    action_dict["sneak"] = batch_sample(action_vec[:, :, 16:18], evaluation)
    action_dict["sprint"] = batch_sample(action_vec[:, :, 18:20], evaluation)
    return action_dict


if __name__ == "__main__":
    print("sample tests")
    dist = [0.25, 0.75]
    counts = np.zeros(2)
    for i in range(100):
        counts += sample(dist, False)
    print(counts)
