import pickle
import random
from typing import Callable, Dict, List

import matplotlib.pyplot as plt

from snake_env import Action, ImprovedSnakeEnv, State


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


def get_best_action(
    state: State, Q_table: Dict[State, Dict[str, float]], actions: List[str]
) -> Action:
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in actions}
    return max(Q_table[state], key=dict_get(Q_table[state]))


def train_q_learning(
    grid_size: int,
    block_size: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    num_episodes: int,
    actions: List[str],
) -> List[float]:
    Q_table: Dict[State, Dict[str, float]] = {}
    env = ImprovedSnakeEnv(
        grid_size=grid_size, block_size=block_size, render_mode="none"
    )
    episode_rewards: List[float] = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = get_best_action(state, Q_table, actions)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if state not in Q_table:
                Q_table[state] = {a: 0 for a in actions}
            if next_state not in Q_table:
                Q_table[next_state] = {a: 0 for a in actions}

            best_next_action = get_best_action(next_state, Q_table, actions)
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

    return episode_rewards


if __name__ == "__main__":
    episode_rewards = train_q_learning(
        grid_size=10,
        block_size=40,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        num_episodes=5000,
        actions=["UP", "DOWN", "LEFT", "RIGHT"],
    )

    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress of Q-Learning Snake Agent")
    plt.savefig("./outputs/training_plot.png")
    plt.show()
