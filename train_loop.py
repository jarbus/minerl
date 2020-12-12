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
