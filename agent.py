# Original code from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
# Copyright 2020 by Jordi TORRES.AI
# Copyright 2022 by Jiwoon Lee(@metr0jw)

from experience import Experience
from env import *

import tensorflow as tf
import numpy as np


class Agent:
    def __init__(self, env, exp_buffer):
        self.state = None
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = tf.constant(state_a)
            q_vals_v = net(state_v)
            act_v = tf.math.reduce_max(q_vals_v)
            action = tf.cast(act_v, tf.int32)

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
