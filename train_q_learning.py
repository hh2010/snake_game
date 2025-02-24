import csv
import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt

from constants import FilePaths, SnakeActions, SnakeConfig, TrainingConfig
from snake_env import ImprovedSnakeEnv, State
from utils import get_best_action


def train_q_learning() -> List[float]:
    Q_table: Dict[State, Dict[str, float]] = {}
    env = ImprovedSnakeEnv(
        grid_size=TrainingConfig.GRID_SIZE,
        block_size=TrainingConfig.BLOCK_SIZE,
        render_mode=SnakeConfig.RENDER_MODE_NONE,
    )
    all_episode_rewards: List[float] = []

    current_epsilon = TrainingConfig.EPSILON_START
    for episode in range(TrainingConfig.NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            if random.uniform(0, 1) < current_epsilon:
                action = random.choice(SnakeActions.all())
            else:
                action = get_best_action(
                    state=state,
                    Q_table=Q_table,
                    actions=SnakeActions.all(),
                    allow_random_on_tie=True,
                )

            next_state, reward, done = env.step(action)
            total_reward += reward

            if state not in Q_table:
                Q_table[state] = {a: 0 for a in SnakeActions.all()}
            if next_state not in Q_table:
                Q_table[next_state] = {a: 0 for a in SnakeActions.all()}

            best_next_action = get_best_action(
                state=next_state,
                Q_table=Q_table,
                actions=SnakeActions.all(),
                allow_random_on_tie=True,
            )
            Q_table[state][action] = Q_table[state][action] + TrainingConfig.ALPHA * (
                reward
                + TrainingConfig.GAMMA * Q_table[next_state][best_next_action]
                - Q_table[state][action]
            )

            state = next_state

            if done:
                break

        if current_epsilon > TrainingConfig.EPSILON_MIN:
            current_epsilon *= TrainingConfig.EPSILON_DECAY

        all_episode_rewards.append(total_reward)

    with open(FilePaths.Q_TABLE_PATH, "wb") as f:
        pickle.dump(Q_table, f)

    with open(
        FilePaths.OUTPUTS_DIR / "training_rewards.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Reward"])
        for episode, reward in enumerate(all_episode_rewards):
            writer.writerow([episode + 1, reward])

    return all_episode_rewards


if __name__ == "__main__":
    training_history = train_q_learning()

    plt.plot(training_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress of Q-Learning Snake Agent")
    plt.savefig(FilePaths.TRAINING_PLOT_PATH)
    plt.show()
