import torch

def J_DQ(pred, actual):
    pass

def J_N(pred, actual):
    pass

def J_E(pred, demo_action_tensor=None, margin=0.8):
    """
    Large Margin Classification Loss
    --------------------------------
    Parameters
      pred:                (b,|A|) tensor of q-values from model
      demo_action_tensor:  (b,|A|) 1-hot vector of expert action
      margin:              int of supervised loss margin

    Returns
      loss:                (1,) tensor of loss magnitude
    """
    if not demo_action_tensor:
        return 0

    # max_a( Q(s,a) - l(s, a_e) ) - Q(s, a_e)
    # creates a (b, 1) tensor
    loss = torch.max(pred + margin*demo_action_tensor,dim=1)\
            - torch.sum(pred * demo_action_tensor,dim=1)

    # (b,|A|) -> (b,1) indicating whether there is
    # a demonstrator action in each row with a 1 or 0
    is_demo = torch.sum(demo_action_tensor,dim=1)

    # masks out losses for non-expert examples
    loss = is_demo * loss

    return torch.sum(loss)/torch.sum(is_demo)

def J_L2(pred, actual):
    pass

def J_Q():
    return J_DQ() + l1*J_N() + l2*J_E() + l3*J_L2()
