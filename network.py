import time
import os
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

import logging

logging.basicConfig()
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Load the MNIST dataset
dataset = fetch_openml("mnist_784", version=1, parser="auto")
images = dataset.data.values.reshape(-1, 28, 28)  # Reshape to 28x28
labels = dataset.target.values


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.sanity_check(n_neurons)
        self.n_neurons = n_neurons
        sqrt = int(np.sqrt(n_neurons))
        self.shape = (sqrt, sqrt)
        self.set_random_pattern()
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        self.patterns = None

    def get_activation(self, i):
        # get activation value of neuron at index i
        weights_to_neuron = self.weights[:, i]
        activation = sum(weights_to_neuron * self.state)
        return activation

    def update(self, i):
        # update neurons at index i and returns if ther neuron was exited
        if self.get_activation(i) >= 0:
            self.state[i] = 1
            return False
        else:
            self.state[i] = -1
            return True

    def run(self, steps):
        # without replacement
        if steps < self.n_neurons:
            indices = np.random.choice(range(self.n_neurons), steps, replace=False)
        else:
            indices = sorted(np.arange(steps), key=lambda k: np.random.random())
        for i in indices:
            while i >= self.n_neurons:
                i -= self.n_neurons
            self.update(i)

    def make_n_changes(self, n_changes):
        if n_changes <= 0:
            return

        n_changes_done = 0
        while n_changes_done != n_changes:
            if self.is_in_local_minima():
                break
            if self.update(np.random.randint(0, self.n_neurons)):
                n_changes_done += 1

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
            raise ValueError(
                f"n_neurons provided is: {n_neurons} but must be at least 4"
            )
        sqrt = np.sqrt(n_neurons)
        if not sqrt.is_integer():
            raise ValueError(
                f"n_neurons provided is: {n_neurons} but must be divisible by itself to an int"
            )
        if n_neurons < 100:
            logger.warning(
                "We recommend to choose n_neurons to be >= 100 to ensure proper generation of characters."
            )

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
            self.patterns = np.concatenate(
                (self.patterns, pattern.reshape(-1, 1)), axis=1
            )

    def get_random_pattern(self):
        return np.random.choice((-1, 1), self.n_neurons)

    def set_random_pattern(self):
        self.state = self.get_random_pattern()

    def set_mutated_pattern(self):
        current = np.copy(self.state)
        for i, state in enumerate(current):
            if np.random.random() < 0.05:
                current[i] = 0 - current[i]
        self.state = current

    def get_number_pattern(self, number):

        # randomly choose an image
        while True:
            i = np.random.randint(len(labels))
            if int(labels[i]) == number:
                break

        arr = images[i].astype(np.uint8)
        img = Image.fromarray(arr)

        resized_img = img.resize(self.shape)
        array = np.array(resized_img)

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
        fig = self.visualize_array(self.weights)
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)
        fig.show()

    def visualize(self, pattern=None):
        if pattern is None:
            pattern = self.state
        pattern = np.reshape(pattern, self.shape)
        return self.visualize_array(pattern)
