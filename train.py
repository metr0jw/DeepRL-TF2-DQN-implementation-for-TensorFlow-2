# Original code from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
# Copyright 2020 by Jordi TORRES.AI
# Copyright 2022 by Jiwoon Lee(@metr0jw)

from env import MaxAndSkipEnv, FireResetEnv, ProcessFrame84, ImageToTensor, BufferWrapper, ScaledFloatFrame
from experience import ExperienceReplay, Experience
from agent import Agent
import models

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import Tensor, summary, constant
import numpy as np
import collections
import datetime
import time
import gym


def make_env(env_name):
    environ = gym.make(env_name)
    environ = MaxAndSkipEnv(environ)
    environ = FireResetEnv(environ)
    environ = ProcessFrame84(environ)
    environ = ImageToTensor(environ)
    # environ = BufferWrapper(environ, 4)
    return ScaledFloatFrame(environ)


DEVICE = "GPU"
FPS = 30
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"  # Name of environment to train
MEAN_REWARD_BOUND = 19.0  # The boundary of reward to stop training

gamma = 0.99  # Discount factor
batch_size = 32  # Minibatch size
replay_size = 10000  # Replay buffer size
learning_rate = 1e-4  # Learning rate
sync_target_frames = 1000  # How frequently sync model weights from main DQN to the target DQN.
replay_start_size = 10000  # Count of frames to add to replay buffer before start training.

eps_start = 1.0  # Hyperparameters related to the epsilon decay schedule
eps_decay = .999985  # Hyperparameters related to the epsilon decay schedule
eps_min = 0.02  # Hyperparameters related to the epsilon decay schedule


print(">>>Training starts at ", datetime.datetime.now())

env = make_env(DEFAULT_ENV_NAME)
input_shape = env.observation_space.shape
n_output = env.action_space.n

net = models.DQN(input_shape, n_output).model
target_net = models.DQN(input_shape, n_output).model
writer = summary.create_file_writer(DEFAULT_ENV_NAME)

state_action_values = 0
expected_state_action_values = 0
net.build(input_shape)

buffer = ExperienceReplay(replay_size)
agent = Agent(env, buffer)

epsilon = eps_start

optimizer = Adam(learning_rate=learning_rate)
mse_loss_fn = MeanSquaredError()
total_rewards = []
frame_idx = 0

best_mean_reward = None


while True:
    frame_idx += 1
    epsilon = max(epsilon * eps_decay, eps_min)

    reward = agent.play_step(net, epsilon)
    if reward is not None:
        total_rewards.append(reward)

        mean_reward = np.mean(total_rewards[-100:])

        print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
            frame_idx, len(total_rewards), mean_reward, epsilon))
        with writer.as_default():
            summary.scalar("epsilon", epsilon, step=frame_idx)
            summary.scalar("reward_100", mean_reward, step=frame_idx)
            summary.scalar("reward", reward, step=frame_idx)
            writer.flush()

        if best_mean_reward is None or best_mean_reward < mean_reward:
            net.save(DEFAULT_ENV_NAME + "-best.h5")
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f" % best_mean_reward)

        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if len(buffer) < replay_start_size:
        continue

    batch = buffer.sample(batch_size)
    states, actions, rewards, dones, next_states = batch

    states_v = Tensor(states)
    next_states_v = Tensor(next_states)
    actions_v = Tensor(actions)
    rewards_v = Tensor(rewards)
    done_mask = constant(dones, dtype='uint8')

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0

    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss_t = mse_loss_fn(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()

    if frame_idx % sync_target_frames == 0:
        target_net.load_state_dict(net.state_dict())

writer.close()
print(">>>Training ends at ", datetime.datetime.now())
