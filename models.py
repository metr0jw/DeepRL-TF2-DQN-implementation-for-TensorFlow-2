from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Model
import numpy as np


class DQN(Model):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv2d1 = Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=input_shape)
        self.conv2d2 = Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv2d3 = Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.dense1 = Dense(512, activation='relu')
        self.out = Dense(n_actions)

    def call(self, inputs):
        conv2d1 = self.conv2d1(inputs)
        conv2d2 = self.conv2d2(conv2d1)
        conv2d3 = self.conv2d3(conv2d2)
        dense1 = self.dense1(conv2d3)
        return self.out(dense1)
