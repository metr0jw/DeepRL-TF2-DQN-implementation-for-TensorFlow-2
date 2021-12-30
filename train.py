from env import MaxAndSkipEnv, FireResetEnv, ProcessFrame84, ImageToTensor, BufferWrapper, ScaledFloatFrame
from agent import Agent
import models
from experience import ExperienceReplay, Experience
import numpy as np
import gym
import gym.spaces


def make_env(env_name):
    environ = gym.make(env_name)
    environ = MaxAndSkipEnv(environ)
    environ = FireResetEnv(environ)
    environ = ProcessFrame84(environ)
    environ = ImageToTensor(environ)
    environ = BufferWrapper(environ, 4)
    return ScaledFloatFrame(environ)


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"     # Name of environment to train
MEAN_REWARD_BOUND = 19.0                    # The boundary of reward to stop training
gamma = 0.99                                # Discount factor
batch_size = 32                             # Minibatch size
replay_size = 10000                         # Replay buffer size
learning_rate = 1e-4                        # Learning rate
sync_target_frames = 1000                   # How frequently sync model weights from main DQN to the target DQN.
replay_start_size = 10000                   # Count of frames to add to replay buffer before start training.
eps_start = 1.0                             # Hyperparameters related to the epsilon decay schedule
eps_decay = .999985                         # Hyperparameters related to the epsilon decay schedule
eps_min = 0.02                              # Hyperparameters related to the epsilon decay schedule


env = make_env(DEFAULT_ENV_NAME)
print(env.action_space.n)
print(env.observation_space.shape)

net = models.DQN(env.observation_space.shape,
                 env.action_space.n).to(device)
target_net = models.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
buffer = ExperienceReplay(replay_size)
agent = Agent(env, buffer)
epsilon = eps_start
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
total_rewards = []
frame_idx = 0
best_mean_reward = None

while True:
    frame_idx += 1
    epsilon = max(epsilon*eps_decay, eps_min)
    reward = agent.play_step(net, epsilon, device=device)
    if reward is not None:
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-100:])
        print("%d: %d games, mean reward %.3f, (epsilon %.2f)" %
              (frame_idx, len(total_rewards), mean_reward, epsilon))
