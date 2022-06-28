# Original code from https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
# Copyright 2020 by Jordi TORRES.AI
# Copyright 2022 by Jiwoon Lee(@metr0jw)

from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.models import Model
import numpy as np


class DQN:
    def __init__(self, input_shape, n_actions):
        self.input = Input(shape=input_shape)
        self.dense1 = Dense(32, activation='relu')(self.input)
        self.dense2 = Dense(64, activation='relu')(self.dense1)
        self.dense3 = Dense(64, activation='relu')(self.dense2)
        self.dense4 = Dense(512, activation='relu')(self.dense3)
        self.out = Dense(n_actions)(self.dense4)

        self.model = Model(self.input, self.out)
