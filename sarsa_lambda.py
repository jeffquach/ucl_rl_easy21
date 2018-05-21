from base_algo import BaseAlgo
from environment import Environment
from monte_carlo import MonteCarlo
from plot_3d import *
import numpy as np

# Question 3: Implemenation of Sarsa(λ) Algo

# Notes:

#     - use same step-size and exploration schedules as in monte_carlo.py
    
#     - use λE{0, 0.1, 0.2, ..., 1}

#     - stop each run after 1000 episodes to report MSE:

#     Σs,a (Q(s,a) - Q*(s,a))^2
#         - over all states s and actions a
#         - Q*(s,a) = action-values computed in monte_carlo.py
#         - Q(s,a) = action-values computed via Sarsa

#     - plot MSE against λ 
    
#     - for λ = 0 and λ = 1 plot learning curve of MSE against episode number

class SarsaLambda(BaseAlgo):

    def __init__(self, env, iterations, lamb, monte_carlo, save_mse_vals=False):
        BaseAlgo.__init__(self, env, iterations)
        self.gamma = 0.8
        self.lamb = lamb
        self.mc = monte_carlo
        if save_mse_vals: 
            self.save_mse_vals = True
            self.mse_vals = []
        else:
            self.save_mse_vals = False

    def mse(self, a, b):
        return np.sum((a - b)**2)

    def train(self):
        for i in range(self.iterations):
            eligibility_trace = np.zeros((10, 21, 2))
            self.env.restart_game()
            state1 = self.env.get_state()
            state1 = (state1[0] - 1, state1[1] - 1)
            action1 = self.epsilon_greedy(state1)

            while state1 != 'terminal':

                state2, reward = self.env.step(action1)
                
                if state2 != 'terminal':
                    state2 = (state2[0] - 1, state2[1] - 1)
                    action2 = self.epsilon_greedy(state2)
                    index2 = state2[0] - 1, state2[1] - 1, action2
                    q_prime = self.action_value_matrix[index2]
                else:
                    action2 = q_prime = 0

                index1 = state1[0] - 1, state1[1] - 1, action1
                
                delta = reward + self.gamma * q_prime - self.action_value_matrix[index1] 
                eligibility_trace[index1] += 1
                self.state_count[index1] += 1

                self.action_value_matrix += 1/self.state_count[index1] * delta * eligibility_trace
                eligibility_trace *= self.gamma * self.lamb

                state1 = state2
                action1 = action2

            if self.save_mse_vals: self.mse_vals.append(self.mse(self.action_value_matrix, self.mc.action_value_matrix))

        return self.mse(self.action_value_matrix, self.mc.action_value_matrix)

if __name__ == "__main__":
    iterations = 50000
    mc = MonteCarlo(Environment(), iterations)
    lambda_values = [round(x * 0.1, 2) for x in range(11)]
    mse_values = []
    lambda_0_and_1 = []
    for l in lambda_values:
        save_mse_vals = True if l == 0.0 or l == 1.0 else False
        sl = SarsaLambda(Environment(), iterations, l, mc, save_mse_vals)
        if save_mse_vals: lambda_0_and_1.append(sl)
        mse_values.append(sl.train())
        plot_3d(11, 22, sl.action_value_matrix, 'sarsa_lambda' + str(l) +'.png')

    line_plot([lambda_values], [mse_values], 'Lambda', 'MSE', 'lambda_vs_mse.png')

    episodes = [i + 1 for i in range(iterations)]

    line_plot([episodes, episodes], [sl.mse_vals for sl in lambda_0_and_1], 'MSE', 'Episodes', 'mse_vs_episodes.png', ['lambda = 0', 'lambda = 1'])