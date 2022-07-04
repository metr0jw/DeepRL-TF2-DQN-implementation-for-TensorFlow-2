# Original code from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
# Copyright 2020 by Jordi TORRES.AI
# Copyright 2022 by Jiwoon Lee(@metr0jw)

import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        exps = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = []
        return np.array(states),\
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)
