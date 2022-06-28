from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Model
import numpy as np


class DQN(Model):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv2a = Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=input_shape)
        self.conv2b = Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv2c = Conv2D(64, kernel_size=3, strides=1, activation='relu')

        self.dense1 = Dense(512, activation='relu')
        self.out = Dense(n_actions)

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = self.conv2b(x)
        x = self.conv2c(x)

        x = self.dense1(x)
        return self.out(x)
