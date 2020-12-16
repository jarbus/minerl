from model import Model
from losses import J_Q
import utils
from utils import *
from utils import Experience
from losses import calcTD
from tqdm import tqdm
import gym
import minerl

class n_step_episode():
    """
    n-step TD: this calss is used to calculate the n-step return and
    bootstrap (if needed) provided a sample tuple
    --------------------------------
    Parameters
      n:                    int;
                            number of steps before bootstrap in n-step TD
      sample:               (5,) tuple;
                            sample of the type ("state", "action", "reward",
                            "next_state", "done")
      is_demo:              bool;
                            defines if the current sample is part of an episode
      gamma:                float in range [0,1);
                            the discount factor

    Returns
      episodSample:         (8,) tensor of loss magnitude
                            sample of the type ("state", "action", "reward",
                            "next_state", "done", "n_step_return", "state_tn",
                            "is_demo")
    """
    def __init__(self, n=10) :
        self.n          = n
        self.steps      = 0
        self.queue      = deque(maxlen=self.n)

    def setup_n_step_tuple(self, sample, is_demo=False, gamma=0.99):
        self.queue.append(sample)
        self.steps += 1
        # push the received samples into the queue until it is full
        if not sample.done and self.steps >= self.n:
            # extract a vector of all rewards
            n_step_return = sum((gamma ** i) * s.reward \
                for s,i in zip(self.queue, range(self.n)))
            tup = Experience(*self.queue.popleft(), \
                n_step_return, self.queue[-1].state, is_demo,None)
            yield tup

        # if length of the episode or the remaining number of
        # samples in the queue are less than n-step then this part
        # is executed
        if sample.done:
            while self.queue:

                n_step_return = sum((gamma ** i) * s.reward \
                    for s,i in zip(self.queue, range(self.n)))
                yield Experience(*self.queue.popleft(), \
                    n_step_return, (torch.zeros((1,3,64,64)), torch.zeros((1,2))), is_demo,None)
            self.steps = 0


def train(replay_buffer,
        behavior_network,
        target_network,
        task=1,
        num_iters=100000,
        batch_size=32,
        tau=10000,
        pre_train_steps=10000,
        n=10,
        gamma=0.99,
        lr=0.01,
        max_ep_len=1000,
        eps_demo=1,
        eps_agent=0.001,
        beta0=0.6):

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
    #print("TRAIN_MASK",TRAINING_MASK)

    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(behavior_network.parameters(), lr=lr)

    # Pretraining
    print("Pre-training...")
    for steps in tqdm(range(pre_train_steps)):

        samples, indices, p_dist, p_sum = replay_buffer.sample(batch_size)
        # Importance Sampling and Priority Replay
        i_s_weights = [(n*p/p_sum)**(-beta0) for p in p_dist]
        max_i_s_weight = max(i_s_weights)
        tds = calcTD(samples, behavior_network,target_network,n=n,gamma=gamma)
        replay_buffer.update_td(indices, tds)

        # Compute weighted loss per sample
        loss = torch.zeros(1)
        for sample, weight, td in zip(samples, i_s_weights, tds):
            loss += (weight / max_i_s_weight) * td * J_Q(target_network, behavior_network, sample)

        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            print("Updated target network")
            target_network.load_state_dict(behavior_network.state_dict())
    print("Pre-training Done.")


    RawExp = namedtuple("RawExp", ["state", "action", "reward", "next_state","done"])
    print("Launching Minecraft...")
    env = gym.make("MineRLNavigateDense-v0")
    n_step_buffer = n_step_episode(n)
    # Training
    obs = env.reset()
    done = False
    net_reward = 0
    it = 0
    for steps in tqdm(range(num_iters)):

        pov, feats = Navigatev0_obs_to_tensor(obs)
        Q_b = behavior_network(pov, feats) * ENV_MASK

        # Turn action tensors into valid Minecraft actions
        # Perform action in Minecraft
        action_dict = action_tensor_to_Navigatev0(Q_b, evaluation=True, task=task)

        obs, reward, done, info = env.step(action_dict)
        it += 1
        if done or it > max_ep_len:
            it = 0
            print("Resetting environment")
            obs = env.reset()
            n_step_buffer = n_step_episode(n)


        state = Navigatev0_obs_to_tensor(obs)
        state_prime = Navigatev0_obs_to_tensor(obs)

        rawexp = RawExp(state, Q_b, torch.tensor(reward,dtype=torch.float32), state_prime, done)
        for exp in n_step_buffer.setup_n_step_tuple(rawexp, is_demo=True):

            td = calcTD([exp], behavior_network,target_network,n=n,gamma=gamma)[0]
            exp = exp._replace(td_error=td)
            replay_buffer.add(exp)


        samples, indices, p_dist, p_sum = replay_buffer.sample(batch_size)
        # Importance Sampling and Priority Replay
        i_s_weights = [(n*p/p_sum)**(-beta0) for p in p_dist]
        max_i_s_weight = max(i_s_weights)
        tds = calcTD(samples, behavior_network,target_network,n=n,gamma=gamma)
        replay_buffer.update_td(indices, tds)

        # Compute weighted loss per sample
        loss = torch.zeros(1)
        for sample, weight, td in zip(samples, i_s_weights, tds):
            loss += (weight / max_i_s_weight) * td * J_Q(target_network, behavior_network, sample)



        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            print("Updated target network")
            target_network.load_state_dict(behavior_network.state_dict())
    return behavior_network, target_network
