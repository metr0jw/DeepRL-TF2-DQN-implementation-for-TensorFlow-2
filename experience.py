# Original code from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
# Copyright 2020 by Jordi TORRES.AI
# Copyright 2022 by Jiwoon Lee(@metr0jw)

import collections
import numpy as np

field_names = ['state', 'action', 'reward', 'done', 'new_state']
Experience = collections.namedtuple('Experience', field_names=field_names)


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
        states = [self.buffer[idx].state for idx in indices]
        actions = [self.buffer[idx].action for idx in indices]
        rewards = [self.buffer[idx].reward for idx in indices]
        dones = [self.buffer[idx].done for idx in indices]
        next_states = [self.buffer[idx].new_state for idx in indices]
        return np.array(states), \
               np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)
