from environment import Environment
from plot_3d import plot_3d
from base_algo import BaseAlgo

# Question 2: Implement Monte-Carlo algo

class MonteCarlo(BaseAlgo):

    def train(self):
        for i in range(self.iterations):
            self.env.restart_game()
            state = self.env.get_state()
            while state != 'terminal':
                state = (state[0] - 1, state[1] - 1)
                action = self.epsilon_greedy(state)
                next_state, reward = self.env.step(action)
                if state != 'terminal':
                    index = state[0] - 1, state[1] - 1, action
                    self.state_count[index] += 1
                    self.action_value_matrix[index] += 1/self.state_count[index] * (reward - self.action_value_matrix[index])
                state = next_state

if __name__ == "__main__":
    mc = MonteCarlo(Environment(), 1000000)
    mc.train()
    plot_3d(11, 22, mc.action_value_matrix, 'monte_carlo.png')