import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.15, gamma=0.98, epsilon=0.25, seed=42):
        self.actions = actions
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q_table = {}
        self.rng = random.Random(seed)

    def choose_action(self, key, explore=True, allowed=None, rng=None):
        rng = self.rng if rng is None else rng
        allowed = list(range(len(self.actions))) if allowed is None else allowed
        values = self.q_table.get(key, [0.0] * len(self.actions))
        if explore and rng.random() < self.epsilon:
            return rng.choice(allowed)
        best = max(values[i] for i in allowed)
        return rng.choice([i for i in allowed if values[i] == best])

    def learn(self, state, action, reward, next_state, done=False, allowed=None):
        values = self.q_table.setdefault(state, [0.0] * len(self.actions))
        next_values = self.q_table.get(next_state, [0.0] * len(self.actions))
        allowed = list(range(len(self.actions))) if allowed is None else allowed
        target = reward if done else reward + self.gamma * max(next_values[i] for i in allowed)
        values[action] += self.alpha * (target - values[action])


