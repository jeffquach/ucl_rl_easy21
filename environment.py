import numpy as np

# Question 1: Implement env simulating Easy21 game
HIT = 0

class Environment:

    def __init__(self):
        self.restart_game()

    def restart_game(self):
        self.player_score = self.dealer_score = 0
        self.player_score += self.draw_from_deck('b')
        self.dealer_score += self.draw_from_deck('b')

    def get_state(self):
        return (self.dealer_score, self.player_score)

    def draw_from_deck(self, colour=None):
        if colour is not None:
            colour = 'black'
        else:
            colour = 'black' if np.random.randint(0,3) > 0 else 'red'

        if colour == 'black':
            return round(np.random.uniform(1,10))
        else:
            return -round(np.random.uniform(1,10))

    def game_over(self, score):
        return score > 21 or score < 1

    def step(self, action):
        if action == HIT:
            self.player_score += self.draw_from_deck()
            if self.game_over(self.player_score):
                next_state, reward = 'terminal', -1
            else:
                next_state, reward = (self.dealer_score, self.player_score), 0
        else:
            while 0 < self.dealer_score < 17:
                self.dealer_score += self.draw_from_deck()
            
            next_state = "terminal"
            if self.game_over(self.dealer_score):
                reward = 1
            elif self.dealer_score > self.player_score:
                reward = -1
            elif self.player_score > self.dealer_score:
                reward = 1
            elif self.dealer_score == self.player_score:
                reward = 0

        return next_state, reward

if __name__ == "__main__":
    # Test Environment implementation
    env = Environment()
    state = None
    for i in range(0, 8):
        print("\nIndex: ", i)
        while state != 'terminal':
            state, reward = env.step(np.random.randint(0, 2))
            print(state, reward)
        env.restart_game()
        state = None