import torch

def J_DQ():
    "For Saber"
    return 0
    pass

def J_N():
    "For Saber/Ryan"
    return 0

def J_E(Q_t, demo_action_tensor=None, margin=0.8):
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
    if not type(demo_action_tensor) or torch.sum(demo_action_tensor).item() == 0:
        return 0

    # max_a( Q(s,a) - l(s, a_e) ) - Q(s, a_e)
    # creates a (b, 1) tensor
    loss = torch.max(Q_t + margin*demo_action_tensor,dim=1)[0]\
            - torch.sum(Q_t * demo_action_tensor,dim=1)

    # (b,|A|) -> (b,1) indicating whether there is
    # a demonstrator action in each row with a 1 or 0
    is_demo = torch.sum(demo_action_tensor,dim=1)
    # masks out losses for non-expert examples
    loss = is_demo * loss

    return torch.sum(loss)/torch.sum(is_demo)

def J_L2(target_network):
    """
    L2 regularization loss. Computed on target network
    """
    l2 = torch.zeros(1,)
    for p in target_network.parameters():
        l2 = l2 + torch.sum(p * p)
    return l2

def J_Q(target_network,
        behavior_network,
        samples,
        l1=1.0,
        l2=1.0,
        l3=1e-5,
        margin=0.8, mask=None):

    states, actions, rewards, is_demo, _  = samples
    Q_t = target_network(states[0],states[1])
    Q_b = behavior_network(states[0],states[1])
    if mask is not None:
        Q_t += mask
        Q_b += mask

    return J_DQ() + l1*J_N() + l2*J_E(Q_t,is_demo,margin=margin) + l3*J_L2(target_network)
