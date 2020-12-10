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
    return F.smooth_l1_loss(Q_b, Q_t, reduction='mean')


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

def J_E(Q_t, demo_action_tensor=None, margin=0.8):
    """
    Large Margin Classification Loss
    --------------------------------
    Parameters
      Q_t:                 (b,|A|) tensor of q-values from model
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
        margin=0.8, 
        mask=None,
        gamma=0.999):

    states, actions, rewards, next_states, n_step_rewards, nth_states, done, is_demo  = samples
    Q_t = target_network(states[0],states[1])
    Q_b = behavior_network(states[0],states[1])

    # to compute the 1-step TD Q-values from target model
    Q_TD = rewards
    # Q_TD[done != 1] += gamma * target_network(n_states[done != 1]).max(dim=1)[0]
    Q_TD += gamma * target_network(next_states).max(dim=1)[0]

    # to compute the n-step TD Q-values from target model
    n = 10
    Q_n = n_step_rewards
    # Q_n[done != 1] += (gamma ** n) * target_network(nth_states[done != 1]).max(dim=1)[0]
    Q_n += (gamma ** n) * target_network(nth_states).max(dim=1)[0]

    if mask is not None:
        Q_t += mask
        Q_b += mask

    return J_DQ(Q_b, Q_TD) + \
           l1*J_n(Q-b, Q_n) + \
           l2*J_E(Q_t,is_demo,margin=margin) + \
           l3*J_L2(target_network)
