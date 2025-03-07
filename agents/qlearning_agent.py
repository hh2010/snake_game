import csv
import datetime
import logging
import os
import pickle
import random
from typing import Dict, Optional

import matplotlib.pyplot as plt

from agents.base_agent import BaseAgent
from constants import (
    Action,
    FilePaths,
    PlotConfig,
    RandomState,
    SnakeActions,
    State,
    TrainingConfig,
)
from snake_env import ImprovedSnakeEnv
from utils import get_best_action
from utils.logging_utils import setup_logger


class QLearningAgent(BaseAgent):
    def __init__(self, debug_mode: bool = False) -> None:
        self.Q_table: Dict[State, Dict[str, float]] = {}
        self.random: random.Random = random.Random(RandomState.TRAINING_SEED)
        self.logger, _ = setup_logger("QLearningAgent", debug_mode)

    @property
    def requires_training(self) -> bool:
        return True

    def choose_action(self, state: State) -> Action:
        if state not in self.Q_table:
            self.Q_table[state] = {a: 0 for a in SnakeActions.all()}
            return self.random.choice(SnakeActions.all())
        max_val = max(self.Q_table[state].values())
        if all(v == max_val for v in self.Q_table[state].values()):
            return self.random.choice(list(self.Q_table[state].keys()))
        return max(self.Q_table[state], key=lambda a: self.Q_table[state][a])

    def _update_q_table(
        self, state: State, action: Action, reward: float, next_state: State
    ) -> None:
        if state not in self.Q_table:
            self.Q_table[state] = {a: 0 for a in SnakeActions.all()}
        if next_state not in self.Q_table:
            self.Q_table[next_state] = {a: 0 for a in SnakeActions.all()}

        best_next: Action = get_best_action(
            next_state, self.Q_table, SnakeActions.all(), self.random
        )
        self.Q_table[state][action] += TrainingConfig.ALPHA * (
            reward
            + TrainingConfig.GAMMA * self.Q_table[next_state][best_next]
            - self.Q_table[state][action]
        )

    def _save_training_data(self, env: ImprovedSnakeEnv, model_name: str) -> None:
        os.makedirs(FilePaths.OUTPUTS_DIR, exist_ok=True)

        rewards_csv_path = os.path.join(
            FilePaths.OUTPUTS_DIR, f"{model_name}_rewards.csv"
        )
        plot_path = os.path.join(FilePaths.OUTPUTS_DIR, f"{model_name}_plot.png")

        episode_rewards = env.get_episode_rewards()

        with open(rewards_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])
            for i, reward in enumerate(episode_rewards):
                writer.writerow([i + 1, reward])

        plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        plt.plot(episode_rewards)
        plt.xlabel(PlotConfig.X_LABEL)
        plt.ylabel(PlotConfig.Y_LABEL)
        plt.title(PlotConfig.TITLE)
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"Training data saved to {rewards_csv_path} and {plot_path}")

    def train(
        self, env: ImprovedSnakeEnv, num_episodes: int, suffix: Optional[str] = None
    ) -> str:
        epsilon: float = TrainingConfig.EPSILON_START

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"{timestamp}_qlearning"

        if suffix:
            model_name = f"{model_name}_{suffix}"

        self.logger.info(f"Starting training with {num_episodes} episodes")
        self.logger.info(f"Initial epsilon: {epsilon}")

        for episode in range(num_episodes):
            state: State = env.reset()

            while True:
                if self.random.uniform(0, 1) < epsilon:
                    action: Action = env.random.choice(SnakeActions.all())
                else:
                    action = get_best_action(
                        state, self.Q_table, SnakeActions.all(), env.random
                    )

                next_state, reward, done = env.step(action)

                self._update_q_table(state, action, reward, next_state)
                state = next_state

                if done:
                    break

            if epsilon > TrainingConfig.EPSILON_MIN:
                epsilon *= TrainingConfig.EPSILON_DECAY

            if (episode + 1) % TrainingConfig.PROGRESS_REPORT_INTERVAL == 0:
                recent_rewards = env.get_episode_rewards()[
                    -TrainingConfig.PROGRESS_REPORT_INTERVAL :
                ]
                avg_reward = (
                    sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                )
                self.logger.info(
                    f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}"
                )

        env.reset()
        self._save_training_data(env, model_name)

        return model_name

    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.Q_table, f)
        self.logger.info(f"Model saved to {filename}")

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            self.Q_table = pickle.load(f)
        self.logger.info(f"Model loaded from {filename}")

    @classmethod
    def create(cls, debug_mode: bool = False) -> "QLearningAgent":
        return cls(debug_mode=debug_mode)
