from base_algo import BaseAlgo
from environment import Environment
from monte_carlo import MonteCarlo
from plot_3d import *
import numpy as np

# Question 4: Linear Function Approximation Algo

# Notes:

    # - use binary feature vector: ϕ(s,a) = 3 * 6 * 2 = 36 features
    #     - Q(s,a) = ϕ(s,a)^T * Θ
    # - use ε = 0.05
    # - use constant step-size = 0.01
    # - plot MSE against λ 
    # - for λ = 0 and λ = 1 plot learning curve of MSE against episode number

    # - (dealer's 1st card, player's sum, action (hit, stick))

class LinearFunctionApproximation():

    def __init__(self, env, iterations, lamb, monte_carlo, save_mse_vals=False):
        self.env = env
        self.iterations = iterations
        self.gamma = 0.8
        self.lamb = lamb
        self.mc = monte_carlo
        self.feature_vector_shape = (3, 6, 2)
        self.feature_vector = {'dealer': [[1, 4], [4, 7], [7, 10]], 'player': [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]}
        self.weights = np.random.random(self.feature_vector_shape) * 0.03
        if save_mse_vals: 
            self.save_mse_vals = True
            self.mse_vals = []
        else:
            self.save_mse_vals = False
        self.epsilon = 0.05
        self.step_size = 0.01
        self.actions = (0, 1)
        self.game_state_shape = (10, 21, 2)

    def action_value_matrix(self):
        q = np.zeros(self.game_state_shape)
        for i in range(self.game_state_shape[0]):
            for j in range(self.game_state_shape[1]):
                for k in range(self.game_state_shape[2]):
                    q[i, j, k] = np.sum(self.binary_feature_vector((i, j), k) * self.weights)
        return q

    def binary_feature_vector(self, state, action):
        dealer, player = state
        vec_money = np.zeros(self.feature_vector_shape)
        for idx, d in enumerate(self.feature_vector['dealer']):
            for ipx, p in enumerate(self.feature_vector['player']):
                if d[0] <= dealer <= d[1] and p[0] <= player <= p[1]:
                    vec_money[idx, ipx, action] = 1
        return vec_money

    # epsilon-greedy algo is taking the argmax of action-value function
    def epsilon_greedy(self, state):
        if state == 'terminal': return 0, None
        if np.random.rand() < (1 - self.epsilon):
            arr = [(np.sum(self.binary_feature_vector(state, action) * self.weights), action) for action in self.actions]
            q, action = arr[0] if arr[0][0] > arr[1][0] else arr[1] 
        else:
            action = np.random.randint(0, 2)
            q = np.sum(self.binary_feature_vector(state, action) * self.weights)
        return q, action

    def mse(self, a, b):
        return np.sum((a - b)**2)

    def train(self):
        for i in range(self.iterations):
            eligibility_trace = np.zeros(self.feature_vector_shape)
            self.env.restart_game()
            state1 = self.env.get_state()
            state1 = (state1[0] - 1, state1[1] - 1)

            while state1 != 'terminal':
                q1, action1 = self.epsilon_greedy(state1)
                state2, reward = self.env.step(action1)

                if state2 != 'terminal':
                    state2 = (state2[0] - 1, state2[1] - 1)
                    q2, action2 = self.epsilon_greedy(state2)
                else:
                    q2 = 0

                delta = reward + self.gamma * q2 - q1
                eligibility_trace = self.gamma * self.lamb * eligibility_trace + self.binary_feature_vector(state1, action1)
                dw = self.step_size * delta * eligibility_trace

                self.weights += dw

                state1 = state2

            if self.save_mse_vals: self.mse_vals.append(self.mse(self.action_value_matrix(), self.mc.action_value_matrix))

        return self.mse(self.action_value_matrix(), self.mc.action_value_matrix)

if __name__ == "__main__":
    iterations = 10000
    mc = MonteCarlo(Environment(), iterations)
    lambda_values = [round(x * 0.1, 2) for x in range(11)]
    mse_values = []
    lambda_0_and_1 = []
    for l in lambda_values:
        save_mse_vals = True if l == 0.0 or l == 1.0 else False
        lfa = LinearFunctionApproximation(Environment(), iterations, l, mc, save_mse_vals)
        if save_mse_vals: lambda_0_and_1.append(lfa)
        mse_values.append(lfa.train())
        plot_3d(11, 22, lfa.action_value_matrix(), 'linear_function_approximation' + str(l) +'.png')

    line_plot([lambda_values], [mse_values], 'Lambda', 'MSE', 'linear_function_approx_lambda_vs_mse.png')

    episodes = [i + 1 for i in range(iterations)]

    line_plot([episodes, episodes], [lfa.mse_vals for lfa in lambda_0_and_1], 'MSE', 'Episodes', 'linear_function_approx_mse_vs_episodes.png', ['lambda = 0', 'lambda = 1'])