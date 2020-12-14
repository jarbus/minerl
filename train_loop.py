from model import Model
from losses import J_Q
import utils
from utils import *
from tqdm import tqdm
import gym
import minerl

# We multiply all actions not used for our task by 0
# We do this by multiplying output vectors by zero
# So weights for a specific action do not get updated


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
        self.experience = namedtuple("Experience", ["state", "action", \
          "reward", "next_state", "done", "n_step_return", "state_tn", "is_demo"])

    def setup_n_step_tuple(self, sample, is_demo=False, gamma=0.99):
        self.queue.append(sample)
        self.steps += 1
        # push the received samples into the queue until it is full
        if not sample.done and self.steps >= self.n:
            # extract a vector of all rewards
            n_step_return = sum((gamma ** i) * s.reward \
                for s,i in zip(self.queue, range(self.n)))
            tup = self.experience(*self.queue.popleft(), \
                n_step_return, self.queue[-1].state, is_demo)
            yield tup

        # if length of the episode or the remaining number of 
        # samples in the queue are less than n-step then this part
        # is executed 
        if sample.done:
            while self.queue:
                n_step_return = sum((gamma ** i) * s.reward \
                    for s,i in zip(self.queue, range(self.n)))
                yield self.experience(*self.queue.popleft(), \
                    n_step_return, None, is_demo)


def train(replay_buffer,
        task=1,
        num_iters=100000,
        batch_size=32,
        tau=1000,
        pre_train_steps=10000,
        lr=0.01):

    """Create action masks.
    An action mask is a (batch_size,11) vector whose values
    are 0 for action indicies in the task and -inf otherwise.
    We can add actions by this mask to ignore actions
    not in our task.
    """
    print(f"Creating action masks for task {task}:")
    print(f"Actions allowed: {TASK_ACTIONS[task]}")
    TRAINING_MASK = torch.full((batch_size,11), 0.0)
    ENV_MASK = torch.full((1,11),0.0)
    for a in TASK_ACTIONS[task]:
        TRAINING_MASK[:,a] = 1.0
        ENV_MASK[:,a] = 1.0

    print("ENV_MASK",ENV_MASK)
    print("TRAIN_MASK",TRAINING_MASK)



    behavior_network = Model()
    target_network = Model()
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.Adam(behavior_network.parameters(), lr=lr)

    # Pretraining
    print("Pre-training...")
    for steps in tqdm(range(pre_train_steps)):
        samples = replay_buffer.sample(batch_size)
        loss = J_Q(target_network, behavior_network, samples)
        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            target_network.load_state_dict(behavior_network.state_dict())

    env = gym.make("MineRLNavigateDense-v0")
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
        if done:
            it = 0
            obs = env.reset()


        state = Navigatev0_obs_to_tensor(obs)
        state_prime = Navigatev0_obs_to_tensor(obs)
        #actions = Navigatev0_action_to_tensor(action_dict,task=task)
        replay_buffer.add(state, Q_b, torch.tensor(reward,dtype=torch.float32), state_prime, done,  0, state_prime, False)

        samples = replay_buffer.sample(batch_size)
        loss = J_Q(target_network, behavior_network, samples)
        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            target_network.load_state_dict(behavior_network.state_dict())
    return behavior_network
