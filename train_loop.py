from model import Model
from losses import J_Q
import utils
from utils import *
from utils import Experience
from losses import calcTD
from tqdm import tqdm
import gym
import minerl
import matplotlib.pyplot as plt
from collections import Counter

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
                n_step_return, self.queue[-1].state, is_demo,None,torch.ones(1))
            yield tup

        # if length of the episode or the remaining number of
        # samples in the queue are less than n-step then this part
        # is executed
        if sample.done:
            while self.queue:

                n_step_return = sum((gamma ** i) * s.reward \
                    for s,i in zip(self.queue, range(self.n)))
                yield Experience(*self.queue.popleft(), \
                    n_step_return, (torch.zeros((1,3,64,64)), torch.zeros((1,2))), is_demo,None, torch.zeros(1))
            self.steps = 0


def train(replay_buffer,
        behavior_network,
        target_network,
        mask,
        task=1,
        num_iters=100000,
        batch_size=32,
        tau=10000,
        pre_train_steps=10000,
        path="",
        n=10,
        gamma=0.99,
        lr=0.01,
        max_ep_len=1000,
        eps_demo=1,
        eps_greedy=0.01,
        eps_agent=0.001,
        lambda3=1e-5,
        beta0=0.6):


    optimizer = torch.optim.Adam(behavior_network.parameters(), lr=lr, weight_decay=lambda3)
    avg_td_errors = []
    avg_loss = []
    pre_train_qs = []
    actions_taken = []

    # Pretraining
    print("Pre-training...")
    for steps in tqdm(range(pre_train_steps)):

        samples, indices, p_dist, p_sum = replay_buffer.sample(batch_size)
        # Importance Sampling and Priority Replay
        i_s_weights = torch.tensor([(n*p/p_sum)**(-beta0) for p in p_dist])
        i_s_weights = i_s_weights/torch.max(i_s_weights)
        i_s_weight_list = i_s_weights.tolist()
        tds = torch.abs(calcTD(samples, behavior_network,target_network,mask,n=n,gamma=gamma))
        replay_buffer.update_td(indices, tds.tolist())
        j_q, Q_t = J_Q(target_network, behavior_network, samples, mask=mask)
        loss = torch.sum(i_s_weights * tds * j_q)
        print(loss.item(),torch.sum(j_q).item())
        avg_td_errors.append(tds.mean().item())
        avg_loss.append(loss.item())
        pre_train_qs.append(Q_t.max(dim=1)[0].mean().item())

        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            print("Updated target network")
            target_network.load_state_dict(behavior_network.state_dict())
    print("Pre-training Done.")
    plt.title("Average Loss over time")
    plt.xlabel("Pre-Training Steps")
    plt.ylabel("Loss")
    plt.plot(avg_loss)
    plt.show()


    plt.title("Average Q_t  over time")
    plt.xlabel("Pre-Training Steps")
    plt.ylabel("Q_t")
    plt.plot(pre_train_qs)
    plt.show()


    plt.title("Average TD  over time")
    plt.xlabel("Pre-Training Steps")
    plt.ylabel("TD")
    plt.plot(avg_td_errors)
    plt.show()



    torch.save(target_network.state_dict(), path)


    RawExp = namedtuple("RawExp", ["state", "action", "reward", "next_state","done"])
    print("Launching Minecraft...")
    env = gym.make("MineRLNavigateDense-v0")
    n_step_buffer = n_step_episode(n)
    # Training
    obs = env.reset()
    done = False
    net_reward = 0
    it = 0
    reward_hist = []
    running_reward_sum = 0
    action_counter = []
    Q_counter = []
    for steps in tqdm(range(num_iters)):

        pov, feats = Navigatev0_obs_to_tensor(obs)
        Q_b = behavior_network(pov, feats) * mask
        action_counter.append(Q_b.max(dim=1)[1].item())
        Q_counter.append(Q_b[Q_b != 0.0].max(dim=1)[0].item())

        # Turn action tensors into valid Minecraft actions
        # Perform action in Minecraft
        action_dict = action_tensor_to_Navigatev0(Q_b, evaluation=False,epsilon=eps_greedy, task=task)

        obs, reward, done, info = env.step(action_dict)
        running_reward_sum += reward

        it += 1
        if done or it > max_ep_len:
            reward_hist.append(running_reward_sum/it)
            print(reward_hist[-1])
            it = 0
            print("Resetting environment")
            obs = env.reset()
            n_step_buffer = n_step_episode(n)


        state = Navigatev0_obs_to_tensor(obs)
        state_prime = Navigatev0_obs_to_tensor(obs)

        rawexp = RawExp(state, Q_b, torch.tensor(reward,dtype=torch.float32), state_prime, done)
        for exp in n_step_buffer.setup_n_step_tuple(rawexp, is_demo=False):


            td = abs(calcTD([exp], behavior_network,target_network,mask=mask,n=n,gamma=gamma)[0].item()) + eps_a
            exp = exp._replace(td_error=td)
            replay_buffer.add(exp)

        # Importance Sampling and Priority Replay
        i_s_weights = torch.tensor([(n*p/p_sum)**(-beta0) for p in p_dist])
        i_s_weights = i_s_weights/torch.max(i_s_weights)
        tds = calcTD(samples, behavior_network,target_network,mask,n=n,gamma=gamma)
        replay_buffer.update_td(indices, torch.abs(tds).tolist())
        loss = torch.sum(i_s_weights * tds * J_Q(target_network, behavior_network, samples, mask=mask))


        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            print("Updated target network")
            target_network.load_state_dict(behavior_network.state_dict())
            torch.save(target_network.state_dict(), path)

    plt.title("Average Reward")
    plt.plot(reward_hist)
    plt.show()
    return behavior_network, target_network
