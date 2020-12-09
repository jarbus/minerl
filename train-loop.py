from model import Model
from losses import J_Q
replay_buffer = ReplayBuffer()
PRE_TRAIN_STEPS = k
BATCH_SIZE = n

# We multiply all actions not used for our task by 0
# We do this by multiplying output vectors by zero
# So weights for a specific action do not get updated


def train(replay_buffer,num_iters=100000,batch_size=32,tau=1000):

    """Create action masks.
    An action mask is a (batch_size,11) vector whose values
    are 0 for action indicies in the task and -inf otherwise.
    We can add actions by this mask to ignore actions
    not in our task.
    """
    TRAINING_MASK = torch.full((BATCH_SIZE,11), -float("Inf"))
    ENV_MASK = torch.full((1,11),-float("Inf"))
    for a in TASK_ACTIONS[TASK]:
        TRAINING_MASK[:,a] = 0.0
        ENV_MASK[:,a] = 0.0



    behavior_network = Model()
    target_network = Model()
    optimizer = torch.optim.Adam(behavior_network.parameters(), lr=LR)

    # Pretraining
    for steps in range(PRE_TRAIN_STEPS):
        samples = replay_buffer.sample(BATCH_SIZE)
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
    for steps in range(num_iters):

        pov, feats = Navigatev0_obs_to_tensor(obs)
        Q_b = behavior_network(pov, feats) + ENV_MASK

        # Turn action tensors into valid Minecraft actions
        # Perform action in Minecraft
        action_dict = action_tensor_to_Navigatev0(outputs[0], evaluation=True, task=TASK)

        obs, reward, done, info = env.step(action_dict)
        if done:
            it = 0
            obs = env.reset()

        replay_buffer.add(prev_state, action_dict, reward, state, done)

        samples = replay_buffer.sample(BATCH_SIZE)
        loss = J_Q(target_network, behavior_network, samples)
        loss.backward()
        optimizer.step()
        if steps % tau == 0:
            target_network.load_state_dict(behavior_network.state_dict())
