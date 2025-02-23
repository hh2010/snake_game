import pickle
import random
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt

from snake_env import Action, ImprovedSnakeEnv, State


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


# Define hyperparameters
alpha: float = 0.1  # Learning rate
gamma: float = 0.9  # Discount factor
epsilon: float = 1.0  # Exploration rate (starts high, decays over time)
epsilon_decay: float = 0.995
epsilon_min: float = 0.01
num_episodes: int = 5000
actions: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]

# Initialize Q-table as a dictionary
Q_table: Dict[State, Dict[str, float]] = {}


def get_best_action(state: State) -> Action:
    """Selects the action with the highest Q-value for a given state."""
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in actions}  # Initialize unseen states with 0

    return max(
        Q_table[state], key=dict_get(Q_table[state])
    )  # Action with highest Q-value


def train_q_learning() -> List[float]:
    global epsilon
    env = ImprovedSnakeEnv(grid_size=10)
    episode_rewards: List[float] = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = get_best_action(state)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if state not in Q_table:
                Q_table[state] = {a: 0 for a in actions}
            if next_state not in Q_table:
                Q_table[next_state] = {a: 0 for a in actions}

            best_next_action = get_best_action(next_state)
            Q_table[state][action] = Q_table[state][action] + alpha * (
                reward
                + gamma * Q_table[next_state][best_next_action]
                - Q_table[state][action]
            )

            state = next_state

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        episode_rewards.append(total_reward)

    with open("./models/q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    return episode_rewards  # Add return statement here


# Train and plot results
episode_rewards = train_q_learning()

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress of Q-Learning Snake Agent")
plt.savefig("./outputs/training_plot.png")
plt.show()
