import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt

from snake_env import ImprovedSnakeEnv, State
from utils import get_best_action


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
    all_episode_rewards: List[float] = []

    current_epsilon = epsilon
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            if random.uniform(0, 1) < current_epsilon:
                action = random.choice(actions)
            else:
                action = get_best_action(
                    Q_table=Q_table,
                    state=state,
                    actions=actions,
                    allow_random_on_tie=True,
                )

            next_state, reward, done = env.step(action)
            total_reward += reward

            if state not in Q_table:
                Q_table[state] = {a: 0 for a in actions}
            if next_state not in Q_table:
                Q_table[next_state] = {a: 0 for a in actions}

            best_next_action = get_best_action(
                Q_table=Q_table,
                state=next_state,
                actions=actions,
                allow_random_on_tie=True,
            )
            Q_table[state][action] = Q_table[state][action] + alpha * (
                reward
                + gamma * Q_table[next_state][best_next_action]
                - Q_table[state][action]
            )

            state = next_state

            if done:
                break

        if current_epsilon > epsilon_min:
            current_epsilon *= epsilon_decay

        all_episode_rewards.append(total_reward)

    with open("./models/q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    return all_episode_rewards


if __name__ == "__main__":
    training_history = train_q_learning(
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

    plt.plot(training_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress of Q-Learning Snake Agent")
    plt.savefig("./outputs/training_plot.png")
    plt.show()
