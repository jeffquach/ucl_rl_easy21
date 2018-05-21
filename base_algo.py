import numpy as np

class BaseAlgo:
    def __init__(self, env, iterations=100):
        self.state_count = np.zeros((10, 21, 2))
        self.action_value_matrix = np.zeros((10, 21, 2))
        self.env = env
        self.iterations = iterations

    def epsilon(self, state):
        return 100 / (100 + np.sum(self.state_count[state]))

    def epsilon_greedy(self, state):
        if np.random.rand() < (1 - self.epsilon(state)):
            return np.argmax(self.action_value_matrix[state])
        else:
            return np.random.randint(0, 2)