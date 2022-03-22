import time
import os
import numpy as np
import torch
import torchvision
from torch import Tensor, nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import logging

logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dataset = torchvision.datasets.MNIST(os.getcwd() + "/files/MNIST/", train=True, download=True)


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.sanity_check(n_neurons)
        self.n_neurons = n_neurons
        sqrt = int(np.sqrt(n_neurons))
        self.shape = (sqrt, sqrt)
        self.state = self.get_random_pattern()
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        self.patterns = None

    def get_activation(self, i):
        # get activation value of neuron at index i
        weights_to_neuron = self.weights[:, i]
        activation = sum(weights_to_neuron * self.state)
        return activation

    def update(self, i):
        # update neurons at index i
        if self.get_activation(i) >= 0:
            self.state[i] = 1
        else:
            self.state[i] = -1

    def run(self, steps):
        # without replacement
        start = time.time()
        if steps < self.n_neurons:
            indices = np.random.choice(range(self.n_neurons), steps, replace=False)
        else:
            indices = sorted(np.arange(steps), key=lambda k: np.random.random())
        for i in indices:
            while i >= self.n_neurons:
                i -= self.n_neurons
            self.update(i)
        logger.debug(f"Made {steps} in {int(time.time() - start)} seconds.")

    def solve(self):
        start = time.time()
        while not self.is_in_local_minima():
            self.run(self.n_neurons)
        logger.debug(f"Solved in {int(time.time() - start)} seconds.")

    def train(self):
        start = time.time()
        i, j = 0, 0
        stop = self.n_neurons - 1
        shown = {}

        while i != stop or j != stop:
            if j < stop:
                j += 1
            elif j == stop:
                j = 0
                i += 1
            if (i, j) in shown or i == j:
                continue
            else:
                shown[(j, i)] = None
                hebb_sum = np.mean(self.patterns[i, :] * self.patterns[j, :])
                self.weights[i, j] = hebb_sum
                self.weights[j, i] = hebb_sum
        logger.debug(f"Trained Network in {int(time.time()-start)} seconds.")

    def is_in_local_minima(self):
        for i in range(self.n_neurons):
            activation = self.get_activation(i)
            if activation >= 0:
                proper_state = 1
            else:
                proper_state = -1
            if self.state[i] != proper_state:
                return False
        return True

    def sanity_check(self, n_neurons):
        if n_neurons < 4:
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be at least 4")
        sqrt = np.sqrt(n_neurons)
        if not sqrt.is_integer():
            raise ValueError(f"n_neurons provided is: {n_neurons} but must be divisible by itself to an int")
        if n_neurons < 100:
            logger.warning("We recommend to choose n_neurons to be >= 100 to ensure proper generation of characters.")

    def is_saved(self, pattern):
        if self.patterns is None:
            return False
        for i in range(self.patterns.shape[1]):
            if (self.patterns[:, i] == pattern).all():
                return True
        return False

    def get_pattern_index(self, pattern):
        for i in range(self.patterns.shape[1]):
            if (self.patterns[:, i] == pattern).all():
                return i
        raise RuntimeError("Pattern is not saved.")

    def save_pattern(self, pattern):
        if self.patterns is None:
            self.patterns = pattern.reshape(-1, 1)
        else:
            self.patterns = np.concatenate((self.patterns, pattern.reshape(-1, 1)), axis=1)
        print("patterns", self.patterns)

    def get_random_pattern(self):
        return np.random.choice((-1, 1), self.n_neurons)

    def get_number_pattern(self, number):
        while True:
            img, n = dataset[np.random.randint(len(dataset))]
            if n == number:
                break
        img = img.resize(self.shape)
        array = np.asarray(img)
        array = np.round(array / 255)
        array[array == 0] = -1
        return array.flatten()

    def visualize_array(self, array):
        fig = Figure(figsize=(3, 3), dpi=100)
        plot = fig.add_subplot(111)
        plot.imshow(array, cmap="Blues", interpolation="nearest")
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        return fig

    def visualize_weight_matrix(self):
        self.visualize_array(self.weights)

    def visualize(self, pattern=None):
        if pattern is None:
            pattern = self.state
        pattern = np.reshape(pattern, self.shape)
        return self.visualize_array(pattern)