import torch
import torch.nn.functional as F


def calcTD(self, sampleB):
    '''
    arguments:
        sampleB: list
            a list of random experiences from the replay memory.

    returns:
    '''

    # to split and concatenate "state", "action", "reward", "next_state", "done"
    # in separate lists
    batch = self.experience(*zip(*sampleB))
    stateB = torch.cat(batch.state)
    actionB = torch.cat(batch.action)
    rewardB = torch.cat(batch.reward)
    next_stateB = torch.cat(batch.next_state)
    doneB = torch.cat(batch.done)

    # given a state s_t to the behavior network, compute Q(s_t)
    # then we use it to calculate Q(s_t, a_t) according to a greedy policy
    Q_behaviorB = self.behavior_net(stateB).gather(1, actionB)

    # to compute the expected target Q-values
    Q_targetB = rewardB
    Q_targetB[doneB != 1] += self.gamma * \
        self.target_net(next_stateB[doneB != 1]).max(1)[0].detach()

    return Q_behaviorB, Q_targetB

def J_DQ(Q_b, Q_TD):
    """
    1-step TD Loss: this function uses Huber function to calculate
    the loss between Q-values from  the behavioral and 1-step TD values
    from the target model outputs.
    --------------------------------
    Parameters
      Q_b:                 (b,|A|) tensor of q-values from behavior model
      Q_TD:                (b,|A|) tensor of TD 1-step q-values from target model

    Returns
      loss:                (1,) tensor of loss magnitude
    """
    return F.smooth_l1_loss(Q_b, Q_TD, reduction='mean')


def J_n(Q_b, Q_n):
    """
    n-step TD Loss: this function uses Huber function to calculate
    the loss between Q-values from  the behavioral and n-step TD values
    from the target model outputs.
    --------------------------------
    Parameters
      Q_b:                 (b,|A|) tensor of q-values from behavior model
      Q_n:                 (b,|A|) tensor of TD n-step q-values from target model

    Returns
      loss:                (1,) tensor of loss magnitude
    """
    return F.smooth_l1_loss(Q_b, Q_n, reduction='mean')

def J_E(Q_t, action_tensor, is_demo, margin=0.8):
    """
    Large Margin Classification Loss
    --------------------------------
    Parameters
      Q_t:                 (b,|A|) tensor of q-values from model
      action_tensor:       (b,|A|) 1-hot vector of actions taken
      is_demo:             (b,1)   binary value indicating if expert action
      margin:              int of supervised loss margin

    Returns
      loss:                (1,) tensor of loss magnitude
    """
    action_tensor = action_tensor.clone().detach().requires_grad_(False)
    # creates a (b, 1) tensor
    loss, _ = torch.max(Q_t + margin*action_tensor,dim=1)
    loss    = loss - torch.sum(Q_t * action_tensor,dim=1)

    denominator = torch.max(torch.ones(1),torch.sum(is_demo))

    # masks out losses for non-expert examples
    print(loss,torch.sum(loss),is_demo, denominator)
    loss = is_demo * loss
    return torch.sum(loss)/denominator

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
        margin=0.8,
        mask=None,
        gamma=0.999):

    povs = torch.cat([s.state[0] for s in samples],dim=0)
    feats = torch.cat([s.state[1] for s in samples],dim=0)
    actions = torch.cat([s.action for s in samples],dim=0)
    rewards = torch.stack([s.reward for s in samples],dim=0)
    next_states_pov = torch.cat([s.next_state[0] for s in samples],dim=0)
    next_states_feats = torch.cat([s.next_state[1] for s in samples],dim=0)
    dones = torch.stack([torch.tensor(s.done) for s in samples],dim=0)
    n_step_returns = torch.stack([torch.tensor(s.n_step_return,dtype=torch.float32) for s in samples],dim=0)
    nth_states_pov = torch.cat([s.state_tn[0] for s in samples],dim=0)
    nth_states_feats = torch.cat([s.state_tn[1] for s in samples],dim=0)
    is_demo = torch.stack([torch.tensor(s.is_demo,dtype=torch.float32) for s in samples],dim=0)

    Q_t = target_network(povs,feats)
    Q_b = behavior_network(povs,feats)

    # to compute the 1-step TD Q-values from target model
    Q_TD = rewards
    # Q_TD[done != 1] += gamma * target_network(n_states[done != 1]).max(dim=1)[0]
    Q_TD += gamma * target_network(next_states_pov,next_states_feats).max(dim=1)[0]

    # to compute the n-step TD Q-values from target model
    n = 10
    Q_n = n_step_returns
    # Q_n[done != 1] += (gamma ** n) * target_network(nth_states[done != 1]).max(dim=1)[0]
    Q_n += (gamma ** n) * target_network(nth_states_pov,nth_states_feats).max(dim=1)[0]

    if mask is not None:
        Q_t *= mask
        Q_b *= mask
    j_dq = J_DQ(Q_b, Q_TD)
    j_n  = l1*J_n(Q_b, Q_n)
    j_e  = l2*J_E(Q_t,actions,is_demo,margin=margin)
    j_l2 = l3*J_L2(target_network)
    return j_dq + j_e + j_l2
